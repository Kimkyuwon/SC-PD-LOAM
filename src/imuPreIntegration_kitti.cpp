#include <math.h>
#include <vector>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/search.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/normal_3d.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <geometry_msgs/TwistStamped.h>
#include <std_msgs/Float32.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <string>

#include "common.h"
#include "tic_toc.h"
#include "pd_loam/frame.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Pose2.h>

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

using namespace std;
using namespace Eigen;

double ProcessTimeMean = 0;
int FrameNum = 0;

using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)

double timelidarOdom;
double timeImu;

sensor_msgs::Imu thisImu;
const double dt = 0.1;

bool systemInitialized = false;
bool imuStart = false;
bool gpsInitialized = false;

Eigen::Matrix3d extRot;
Eigen::Vector3d gpsXYZ0;
Eigen::Quaterniond currQ;
std::queue<pd_loam::frameConstPtr> FrameBuf;
std::mutex mBuf;
std::mutex mFrame;

gtsam::PreintegratedImuMeasurements *imuIntegratorOpt_;
gtsam::PreintegratedImuMeasurements *imuIntegratorImu_;

gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;
gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
gtsam::noiseModel::Diagonal::shared_ptr gpsNoise;
gtsam::noiseModel::Diagonal::shared_ptr gpsVelNoise;
gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;
gtsam::noiseModel::Diagonal::shared_ptr correctionNoise2;
gtsam::Vector noiseModelBetweenBias;
float imuAccBiasN, imuGyrBiasN, imuAccNoise, imuGyrNoise, imuGravity;

gtsam::ISAM2 optimizer;
gtsam::NonlinearFactorGraph graphFactors;
gtsam::Values graphValues;
int key = 1;

gtsam::Pose3 lidarPose;
gtsam::Pose3 prevPose_;
gtsam::Vector3 prevVel_;
gtsam::NavState prevState_;
gtsam::imuBias::ConstantBias prevBias_;

gtsam::NavState prevStateOdom;
gtsam::imuBias::ConstantBias prevBiasOdom;

double gpsAlt = 0;
gtsam::Pose3 gpsVel;

nav_msgs::Path gpsPath, imuPath, laserPath;

ros::Publisher pubImuOdometry, pubGpsPath, pubImuPath, pubLaserPath, pubLaserOdom, pubFrameAftImu;
ros::Publisher pubProcessTime;

Eigen::Affine3f odom2affine(nav_msgs::Odometry odom)
{
    double x, y, z, roll, pitch, yaw;
    x = odom.pose.pose.position.x;
    y = odom.pose.pose.position.y;
    z = odom.pose.pose.position.z;
    tf::Quaternion orientation;
    tf::quaternionMsgToTF(odom.pose.pose.orientation, orientation);
    tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
    return pcl::getTransformation(x, y, z, roll, pitch, yaw);
}

void resetOptimization()
{
    gtsam::ISAM2Params optParameters;
    optParameters.relinearizeThreshold = 0.1;
    optParameters.relinearizeSkip = 1;
    optimizer = gtsam::ISAM2(optParameters);

    gtsam::NonlinearFactorGraph newGraphFactors;
    graphFactors = newGraphFactors;

    gtsam::Values NewGraphValues;
    graphValues = NewGraphValues;
}

bool failureDetection(const gtsam::Vector3& velCur, const gtsam::imuBias::ConstantBias& biasCur)
{
    Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z());
    if (vel.norm() > 50)
    {
        ROS_WARN("Large velocity, reset IMU-preintegration!");
        return true;
    }

    Eigen::Vector3f ba(biasCur.accelerometer().x(), biasCur.accelerometer().y(), biasCur.accelerometer().z());
    Eigen::Vector3f bg(biasCur.gyroscope().x(), biasCur.gyroscope().y(), biasCur.gyroscope().z());
    if (ba.norm() > 1.0 || bg.norm() > 1.0)
    {
        ROS_WARN("Large bias, reset IMU-preintegration!");
        return true;
    }

    return false;
}

void resetParams()
{
    systemInitialized = false;
}

sensor_msgs::Imu imuConverter(const sensor_msgs::Imu& imu_in)
{
    sensor_msgs::Imu imu_out = imu_in;
    // rotate acceleration
    Eigen::Vector3d acc(imu_in.linear_acceleration.x, imu_in.linear_acceleration.y, imu_in.linear_acceleration.z);
    acc = extRot * acc;
    imu_out.linear_acceleration.x = acc.x();
    imu_out.linear_acceleration.y = acc.y();
    imu_out.linear_acceleration.z = acc.z();
    // rotate gyroscope
    Eigen::Vector3d gyr(imu_in.angular_velocity.x, imu_in.angular_velocity.y, imu_in.angular_velocity.z);
    gyr = extRot * gyr;
    imu_out.angular_velocity.x = gyr.x();
    imu_out.angular_velocity.y = gyr.y();
    imu_out.angular_velocity.z = gyr.z();
//    // rotate roll pitch yaw
//    Eigen::Quaterniond q_from(imu_in.orientation.w, imu_in.orientation.x, imu_in.orientation.y, imu_in.orientation.z);
//    Eigen::Quaterniond q_final = q_from * extQRPY;
//    imu_out.orientation.x = q_final.x();
//    imu_out.orientation.y = q_final.y();
//    imu_out.orientation.z = q_final.z();
//    imu_out.orientation.w = q_final.w();

//    if (sqrt(q_final.x()*q_final.x() + q_final.y()*q_final.y() + q_final.z()*q_final.z() + q_final.w()*q_final.w()) < 0.1)
//    {
//        ROS_ERROR("Invalid quaternion, please use a 9-axis IMU!");
//        ros::shutdown();
//    }

    return imu_out;
}

void imuHandler(const sensor_msgs::ImuConstPtr &imu)
{
    thisImu = *imu;
}

void laserOdomHandler(const nav_msgs::OdometryConstPtr &odomMsg)
{
    mBuf.lock();
    float p_x = odomMsg->pose.pose.position.x;
    float p_y = odomMsg->pose.pose.position.y;
    float p_z = odomMsg->pose.pose.position.z;
    float r_x = odomMsg->pose.pose.orientation.x;
    float r_y = odomMsg->pose.pose.orientation.y;
    float r_z = odomMsg->pose.pose.orientation.z;
    float r_w = odomMsg->pose.pose.orientation.w;
    lidarPose = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));
    if (systemInitialized == false)
    {
        resetOptimization();
        prevPose_ = lidarPose;
        gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise);
        graphFactors.add(priorPose);
        // initial velocity
        prevVel_ = gtsam::Vector3(0, 0, 0);
        gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, priorVelNoise);
        graphFactors.add(priorVel);
        // initial bias
        prevBias_ = gtsam::imuBias::ConstantBias();
        gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, priorBiasNoise);
        graphFactors.add(priorBias);
        // add values
        graphValues.insert(X(0), prevPose_);
        graphValues.insert(V(0), prevVel_);
        graphValues.insert(B(0), prevBias_);
        // optimize once
        optimizer.update(graphFactors, graphValues);
        graphFactors.resize(0);
        graphValues.clear();

        imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
        imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);

        key = 1;
        systemInitialized = true;
    }

    // reset graph for speed
    if (key == 100)
    {
        // get updated noise before reset
        gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(X(key-1)));
        gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise  = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(V(key-1)));
        gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(B(key-1)));
        // reset graph
        resetOptimization();
        // add pose
        gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
        graphFactors.add(priorPose);
        // add velocity
        gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, updatedVelNoise);
        graphFactors.add(priorVel);
        // add bias
        gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, updatedBiasNoise);
        graphFactors.add(priorBias);
        // add values
        graphValues.insert(X(0), prevPose_);
        graphValues.insert(V(0), prevVel_);
        graphValues.insert(B(0), prevBias_);
        // optimize once
        optimizer.update(graphFactors, graphValues);
        graphFactors.resize(0);
        graphValues.clear();

        key = 1;
    }

    TicToc t_whole;

    imuIntegratorOpt_->integrateMeasurement(
            gtsam::Vector3(thisImu.linear_acceleration.x, thisImu.linear_acceleration.y, thisImu.linear_acceleration.z),
            gtsam::Vector3(thisImu.angular_velocity.x,    thisImu.angular_velocity.y,    thisImu.angular_velocity.z), dt);


    // add imu factor to graph
    const gtsam::PreintegratedImuMeasurements& preint_imu = dynamic_cast<const gtsam::PreintegratedImuMeasurements&>(*imuIntegratorOpt_);
    gtsam::ImuFactor imu_factor(X(key - 1), V(key - 1), X(key), V(key), B(key - 1), preint_imu);
    graphFactors.add(imu_factor);
    // add imu bias between factor
    graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(key - 1), B(key), gtsam::imuBias::ConstantBias(),
                     gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias)));
    // add pose factor
    gtsam::Pose3 curPose = lidarPose;
    gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key), curPose, correctionNoise);
    graphFactors.add(pose_factor);
    //add GPS factor
    gtsam::Point3 gpsConstraint(lidarPose.x(), lidarPose.y(), gpsAlt); // in this example, only adjusting altitude (for x and y, very big noises are set)
    graphFactors.add(gtsam::GPSFactor(X(key), gpsConstraint, gpsNoise));
    graphFactors.add(gtsam::BetweenFactor<gtsam::Pose3>(X(key - 1), X(key), gpsVel, gpsVelNoise));
    // insert predicted values
    gtsam::NavState propState_ = imuIntegratorOpt_->predict(prevState_, prevBias_);
    graphValues.insert(X(key), propState_.pose());
    graphValues.insert(V(key), propState_.v());
    graphValues.insert(B(key), prevBias_);
    // optimize
    optimizer.update(graphFactors, graphValues);
    optimizer.update();
    graphFactors.resize(0);
    graphValues.clear();
    // Overwrite the beginning of the preintegration for the next step.
    gtsam::Values result = optimizer.calculateEstimate();
    prevPose_  = result.at<gtsam::Pose3>(X(key));
    prevVel_   = result.at<gtsam::Vector3>(V(key));
    prevState_ = gtsam::NavState(prevPose_, prevVel_);
    prevBias_  = result.at<gtsam::imuBias::ConstantBias>(B(key));
    // Reset the optimization preintegration object.
    imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);

    // predict odometry
    gtsam::NavState imuState = imuIntegratorOpt_->predict(prevStateOdom, prevBiasOdom);
    // publish odometry
    gtsam::Pose3 ImuIntegPose = gtsam::Pose3(imuState.quaternion(), imuState.position());

    // publish latest odometry
    nav_msgs::Odometry laserOdometry;
    laserOdometry.header.stamp = thisImu.header.stamp;
    laserOdometry.header.frame_id = "/body";
    laserOdometry.child_frame_id = "/lidar_aft_imu";
    laserOdometry.pose.pose.position.x = prevPose_.x();
    laserOdometry.pose.pose.position.y = prevPose_.y();
    laserOdometry.pose.pose.position.z = prevPose_.z();
    laserOdometry.pose.pose.orientation.w = prevPose_.rotation().toQuaternion().w();
    laserOdometry.pose.pose.orientation.x = prevPose_.rotation().toQuaternion().x();
    laserOdometry.pose.pose.orientation.y = prevPose_.rotation().toQuaternion().y();
    laserOdometry.pose.pose.orientation.z = prevPose_.rotation().toQuaternion().z();
    pubLaserOdom.publish(laserOdometry);

    geometry_msgs::PoseStamped laserPoseStamped;
    laserPoseStamped.header = laserOdometry.header;
    laserPoseStamped.pose = laserOdometry.pose.pose;

    laserPath.header.stamp = laserOdometry.header.stamp;
    laserPath.header.frame_id = "/body";
    laserPath.poses.push_back(laserPoseStamped);
    pubLaserPath.publish(laserPath);

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(laserOdometry.pose.pose.position.x, laserOdometry.pose.pose.position.y, laserOdometry.pose.pose.position.z));
    q.setW(laserOdometry.pose.pose.orientation.w);
    q.setX(laserOdometry.pose.pose.orientation.x);
    q.setY(laserOdometry.pose.pose.orientation.y);
    q.setZ(laserOdometry.pose.pose.orientation.z);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, laserOdometry.header.stamp, "/body", "/aft_imu"));

    while(!FrameBuf.empty())
    {
        mFrame.lock();
        pd_loam::frame FrameAftImu;
        FrameAftImu = *FrameBuf.front();
        FrameAftImu.pose = laserOdometry;
        pubFrameAftImu.publish(FrameAftImu);
        FrameBuf.pop();
        mFrame.unlock();
    }

    // check optimization
    if (failureDetection(prevVel_, prevBias_))
    {
        resetParams();
        mBuf.unlock();
        return;
    }

    // 2. after optiization, re-propagate imu odometry preintegration
    prevStateOdom = prevState_;
    prevBiasOdom  = prevBias_;

    double wholeProcessTime = t_whole.toc();
    ProcessTimeMean = ProcessTimeMean*FrameNum + wholeProcessTime;
    FrameNum++;
    ProcessTimeMean /= FrameNum;

    std_msgs::Float32 float_time;
    float_time.data = (float)ProcessTimeMean;
    pubProcessTime.publish(float_time);

    ++key;
    mBuf.unlock();
    FrameNum++;
}

void FrameHandler(const pd_loam::frameConstPtr &frame)
{
    mBuf.lock();
    FrameBuf.push(frame);
    mBuf.unlock();
}

void gpsHandler(const sensor_msgs::NavSatFixConstPtr &gps)
{
    Eigen::Vector3d llh;
    llh(0) = gps->latitude;
    llh(1) = gps->longitude;
    llh(2) = gps->altitude;
    Eigen::Vector3d gpsXYZ = llh2xyz(llh);

    if (gpsInitialized == false)
    {
        gpsXYZ0 = gpsXYZ;
        gpsInitialized = true;
    }
    Eigen::Vector3d gpsENU = xyz2enu(gpsXYZ, gpsXYZ0);
    gpsAlt = gpsENU(2);
}

void gpsVelHandler(const geometry_msgs::TwistStampedConstPtr &gpsvel)
{
    gpsVel = gtsam::Pose3(gtsam::Rot3::RzRyRx(gpsvel->twist.angular.x*0.1, gpsvel->twist.angular.y*0.1, gpsvel->twist.angular.z*0.1),
                          gtsam::Point3(gpsvel->twist.linear.x*0.1, gpsvel->twist.linear.y*0.1, gpsvel->twist.linear.z*0.1));

//    if (laserPath.poses.size() > 1)
//    {
//        int k = laserPath.poses.size() - 1;
//        Eigen::Quaterniond q_curr;
//        q_curr.w() = laserPath.poses[k].pose.orientation.w;
//        q_curr.x() = laserPath.poses[k].pose.orientation.x;
//        q_curr.y() = laserPath.poses[k].pose.orientation.y;
//        q_curr.z() = laserPath.poses[k].pose.orientation.z;
//        Eigen::Matrix3d R_curr = q_curr.toRotationMatrix();
//        Eigen::Matrix4d TF_curr(Eigen::Matrix4d::Identity());
//        TF_curr.block(0,0,3,3) = R_curr;
//        TF_curr(0,3) = laserPath.poses[k].pose.position.x;
//        TF_curr(1,3) = laserPath.poses[k].pose.position.y;
//        TF_curr(2,3) = laserPath.poses[k].pose.position.z;
//        Eigen::Quaterniond q_prev;
//        q_prev.w() = laserPath.poses[k-1].pose.orientation.w;
//        q_prev.x() = laserPath.poses[k-1].pose.orientation.x;
//        q_prev.y() = laserPath.poses[k-1].pose.orientation.y;
//        q_prev.z() = laserPath.poses[k-1].pose.orientation.z;
//        Eigen::Matrix3d R_prev = q_prev.toRotationMatrix();
//        Eigen::Matrix4d TF_prev(Eigen::Matrix4d::Identity());
//        TF_prev.block(0,0,3,3) = R_prev;
//        TF_prev(0,3) = laserPath.poses[k-1].pose.position.x;
//        TF_prev(1,3) = laserPath.poses[k-1].pose.position.y;
//        TF_prev(2,3) = laserPath.poses[k-1].pose.position.z;
//        Eigen::Matrix4d TF_delta = TF_prev.inverse() * TF_curr;
//        std::cout<<"gps : "<<gpsVel.transpose()<<", odom : "<<TF_delta(0,3)<<" "<<TF_delta(1,3)<<" "<<TF_delta(2,3)<<std::endl;
//    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "imuPreIntegration_kitti");
    ros::NodeHandle nh;

    std::string ImuTopic;
    nh.param<string>("imu_topic", ImuTopic, "/kitti/oxts/imu");
    ros::Subscriber subImu = nh.subscribe<sensor_msgs::Imu>(ImuTopic, 100, imuHandler);

    ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/mapping/aft_mapped_to_init", 100, laserOdomHandler);

    ros::Subscriber subFrame = nh.subscribe<pd_loam::frame>("/mapping/frame", 100, FrameHandler);

    ros::Subscriber subGPS = nh.subscribe<sensor_msgs::NavSatFix>("/kitti/oxts/gps/fix", 100, gpsHandler);

    ros::Subscriber subGPSVel = nh.subscribe<geometry_msgs::TwistStamped>("/kitti/oxts/gps/vel", 100, gpsVelHandler);

    pubImuOdometry = nh.advertise<nav_msgs::Odometry>("/imuPreintegration/imu_odom", 100);

    pubGpsPath = nh.advertise<nav_msgs::Path>("/imuPreintegration/gps_path", 100);

    pubImuPath = nh.advertise<nav_msgs::Path>("/imuPreintegration/imu_path", 100);

    pubLaserPath = nh.advertise<nav_msgs::Path>("/imuPreintegration/lidar_aft_imu_path", 100);

    pubLaserOdom = nh.advertise<nav_msgs::Odometry>("/imuPreintegration/lidar_aft_imu_odom", 100);

    pubFrameAftImu = nh.advertise<pd_loam::frame>("/imuPreintegration/frame", 100);

    pubProcessTime = nh.advertise<std_msgs::Float32>("/imu_process_time", 100);

    nh.param<float>("accBias", imuAccBiasN, 0.0002);
    nh.param<float>("gyroBias", imuGyrBiasN, 2.0e-5);
    nh.param<float>("accNoise", imuAccNoise, 0.01);
    nh.param<float>("gyroNoise", imuGyrNoise, 0.001);
    nh.param<float>("gravity", imuGravity, 9.80511);

//    Eigen::AngleAxisd rotX_imu(-M_PI/2, Eigen::Vector3d::UnitX());
//    Eigen::AngleAxisd rotY_imu(0, Eigen::Vector3d::UnitY());
//    Eigen::AngleAxisd rotZ_imu(M_PI/2, Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd rotX_imu(0, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd rotY_imu(0, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd rotZ_imu(0, Eigen::Vector3d::UnitZ());
    extRot = (rotZ_imu * rotY_imu * rotX_imu).matrix();

    boost::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU(imuGravity);
    p->accelerometerCovariance  = gtsam::Matrix33::Identity(3,3) * pow(imuAccNoise, 2); // acc white noise in continuous
    p->gyroscopeCovariance      = gtsam::Matrix33::Identity(3,3) * pow(imuGyrNoise, 2); // gyro white noise in continuous
    p->integrationCovariance    = gtsam::Matrix33::Identity(3,3) * pow(1e-2, 2); // error committed in integrating position from velocities
    gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());; // assume zero initial bias

    priorPoseNoise  = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1).finished()); // rad,rad,rad,m, m, m
    priorVelNoise   = gtsam::noiseModel::Isotropic::Sigma(3, 1e2); // m/s
    priorBiasNoise  = gtsam::noiseModel::Isotropic::Sigma(6, 1e-2); // 1e-2 ~ 1e-3 seems to be good
    correctionNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1).finished()); // rad,rad,rad,m, m, m
    noiseModelBetweenBias = (gtsam::Vector(6) << imuAccBiasN, imuAccBiasN, imuAccBiasN, imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished();
    gpsNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(3) << 10000, 10000, 0.01).finished());
    gpsVelNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 10000, 10000, 0.1, 0.1, 0.1, 0.1).finished());

    imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for IMU message thread
    imuIntegratorOpt_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for optimization

    ros::spin();

    return 0;
}
