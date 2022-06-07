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

using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)

double timelidarOdom;
double timeImu;
double lastImuTime = -1;

bool systemInitialized = false;
bool doneFirstOpt = false;
bool imuStart = false;

Eigen::Matrix3d extRot;
Eigen::Quaterniond extQRPY;

std::queue<sensor_msgs::Imu> imuBuf;
std::queue<sensor_msgs::Imu> imuQueBuf;
std::queue<double> dtBuf;
std::queue<double> dtQueBuf;
std::queue<nav_msgs::OdometryConstPtr> laserOdomBuf;
std::mutex mBuf;

gtsam::PreintegratedImuMeasurements *imuIntegratorOpt_;
gtsam::PreintegratedImuMeasurements *imuIntegratorImu_;

gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;
gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;
gtsam::noiseModel::Diagonal::shared_ptr correctionNoise2;
gtsam::Vector noiseModelBetweenBias;
float imuAccBiasN, imuGyrBiasN, imuAccNoise, imuGyrNoise, imuGravity;

gtsam::ISAM2 optimizer;
gtsam::NonlinearFactorGraph graphFactors;
gtsam::Values graphValues;
int key = 1;

gtsam::Pose3 prevPose_;
gtsam::Vector3 prevVel_;
gtsam::NavState prevState_;
gtsam::imuBias::ConstantBias prevBias_;

gtsam::NavState prevStateOdom;
gtsam::imuBias::ConstantBias prevBiasOdom;

nav_msgs::Path imuPath;

ros::Publisher pubImuOdometry, pubImuPath, pubImuIncremental;

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
    doneFirstOpt = false;
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
    if (imuStart == false)
    {
        lastImuTime = imu->header.stamp.toSec();
        imuStart = true;
        return;
    }
    double dt = imu->header.stamp.toSec() - lastImuTime;
    if (dt < 0.01 || dt > 0.02)
    {
        dt = 0.0162;
    }
    lastImuTime = imu->header.stamp.toSec();
    sensor_msgs::Imu thisImu = imuConverter(*imu);

    mBuf.lock();
    imuBuf.push(thisImu);
    imuQueBuf.push(thisImu);
    dtBuf.push(dt);
    dtQueBuf.push(dt);
    mBuf.unlock();

    if (doneFirstOpt == false)        return;
    // integrate this single imu message
    imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu.linear_acceleration.x, thisImu.linear_acceleration.y, thisImu.linear_acceleration.z),
                                            gtsam::Vector3(thisImu.angular_velocity.x,    thisImu.angular_velocity.y,    thisImu.angular_velocity.z), dt);

    // predict odometry
    gtsam::NavState currentState = imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom);
    // publish odometry
    nav_msgs::Odometry odometry;
    odometry.header.stamp = thisImu.header.stamp;
    odometry.header.frame_id = "/body";
    odometry.child_frame_id = "/odom_imu";

    gtsam::Pose3 imuPose = gtsam::Pose3(currentState.quaternion(), currentState.position());

    odometry.pose.pose.position.x = imuPose.translation().x();
    odometry.pose.pose.position.y = imuPose.translation().y();
    odometry.pose.pose.position.z = imuPose.translation().z();
    odometry.pose.pose.orientation.x = imuPose.rotation().toQuaternion().x();
    odometry.pose.pose.orientation.y = imuPose.rotation().toQuaternion().y();
    odometry.pose.pose.orientation.z = imuPose.rotation().toQuaternion().z();
    odometry.pose.pose.orientation.w = imuPose.rotation().toQuaternion().w();

    odometry.twist.twist.linear.x = currentState.velocity().x();
    odometry.twist.twist.linear.y = currentState.velocity().y();
    odometry.twist.twist.linear.z = currentState.velocity().z();
    odometry.twist.twist.angular.x = thisImu.angular_velocity.x + prevBiasOdom.gyroscope().x();
    odometry.twist.twist.angular.y = thisImu.angular_velocity.y + prevBiasOdom.gyroscope().y();
    odometry.twist.twist.angular.z = thisImu.angular_velocity.z + prevBiasOdom.gyroscope().z();
    pubImuOdometry.publish(odometry);

    geometry_msgs::PoseStamped imuPoseStamped;
    imuPoseStamped.header = odometry.header;
    imuPoseStamped.pose = odometry.pose.pose;

    imuPath.header.stamp = odometry.header.stamp;
    imuPath.header.frame_id = "/body";
    imuPath.poses.push_back(imuPoseStamped);
    pubImuPath.publish(imuPath);

}

void laserOdomHandler(const nav_msgs::OdometryConstPtr &odom)
{
    mBuf.lock();
    laserOdomBuf.push(odom);
    mBuf.unlock();
}

void processImu()
{
    while(1)
    {
        while (!laserOdomBuf.empty())
        {
            mBuf.lock();
            if (laserOdomBuf.empty())
            {
                mBuf.unlock();
                break;
            }
            timelidarOdom = laserOdomBuf.front()->header.stamp.toSec();

            while(imuBuf.front().header.stamp.toSec() < timelidarOdom)
            {
                imuBuf.pop();
                dtBuf.pop();
                imuQueBuf.pop();
                dtQueBuf.pop();
            }

            nav_msgs::OdometryConstPtr odomMsg = laserOdomBuf.front();
            float p_x = odomMsg->pose.pose.position.x;
            float p_y = odomMsg->pose.pose.position.y;
            float p_z = odomMsg->pose.pose.position.z;
            float r_x = odomMsg->pose.pose.orientation.x;
            float r_y = odomMsg->pose.pose.orientation.y;
            float r_z = odomMsg->pose.pose.orientation.z;
            float r_w = odomMsg->pose.pose.orientation.w;
            gtsam::Pose3 lidarPose = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));

            laserOdomBuf.pop();

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

            while (!imuBuf.empty() && !dtBuf.empty())
            {
                sensor_msgs::Imu thisImu = imuBuf.front();
                double dt = dtBuf.front();

                imuIntegratorOpt_->integrateMeasurement(
                        gtsam::Vector3(thisImu.linear_acceleration.x, thisImu.linear_acceleration.y, thisImu.linear_acceleration.z),
                        gtsam::Vector3(thisImu.angular_velocity.x,    thisImu.angular_velocity.y,    thisImu.angular_velocity.z), dt);

                imuBuf.pop();
                dtBuf.pop();
            }

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

            // check optimization
            if (failureDetection(prevVel_, prevBias_))
            {
                resetParams();
                mBuf.unlock();
                continue;
            }

            // 2. after optiization, re-propagate imu odometry preintegration
            prevStateOdom = prevState_;
            prevBiasOdom  = prevBias_;

//            cout<<"bias : "<<prevBias_.vector().transpose()<<endl;
//            cout<<"state : "<<prevState_.t().transpose()<<" "<<prevVel_.transpose()<<endl;

            // repropogate
            if (!imuQueBuf.empty() && !dtQueBuf.empty())
            {
                // reset bias use the newly optimized bias
                imuIntegratorImu_->resetIntegrationAndSetBias(prevBiasOdom);
                // integrate imu message from the beginning of this optimization
                while(!imuQueBuf.empty() && !dtQueBuf.empty())
                {
                    sensor_msgs::Imu thisImu = imuQueBuf.front();
                    double dt = dtQueBuf.front();

                    imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu.linear_acceleration.x, thisImu.linear_acceleration.y, thisImu.linear_acceleration.z),
                                                            gtsam::Vector3(thisImu.angular_velocity.x,    thisImu.angular_velocity.y,    thisImu.angular_velocity.z), dt);
                    imuQueBuf.pop();
                    dtQueBuf.pop();
                }
            }

            // predict odometry
            gtsam::NavState imuState = imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom);
            // publish odometry
            gtsam::Pose3 PrevPose = gtsam::Pose3(prevStateOdom.quaternion(), prevStateOdom.position());
            gtsam::Pose3 ImuIntegPose = gtsam::Pose3(imuState.quaternion(), imuState.position());
            gtsam::Pose3 relPose = PrevPose.between(ImuIntegPose);
            nav_msgs::Odometry imuIncremental;
            imuIncremental.header.stamp = ros::Time().fromSec(timelidarOdom);
            imuIncremental.header.frame_id = "/body";
            imuIncremental.pose.pose.position.x = relPose.x();
            imuIncremental.pose.pose.position.y = relPose.y();
            imuIncremental.pose.pose.position.z = relPose.z();
            imuIncremental.pose.pose.orientation.w = relPose.rotation().toQuaternion().w();
            imuIncremental.pose.pose.orientation.x = relPose.rotation().toQuaternion().x();
            imuIncremental.pose.pose.orientation.y = relPose.rotation().toQuaternion().y();
            imuIncremental.pose.pose.orientation.z = relPose.rotation().toQuaternion().z();
            pubImuIncremental.publish(imuIncremental);

            ++key;
            doneFirstOpt = true;
            mBuf.unlock();
        }

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "imuPreIntegration");
    ros::NodeHandle nh;

    ros::Subscriber subImu = nh.subscribe<sensor_msgs::Imu>("/camera/imu", 100, imuHandler);
    ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/mapping/aft_mapped_to_init", 100, laserOdomHandler);

    pubImuOdometry = nh.advertise<nav_msgs::Odometry>("/imu_odom", 100);

    pubImuPath = nh.advertise<nav_msgs::Path>("/imu_path", 100);

    pubImuIncremental = nh.advertise<nav_msgs::Odometry>("/imu_incremental", 100);

    nh.param<float>("accBias", imuAccBiasN, 0.0002);
    nh.param<float>("gyroBias", imuGyrBiasN, 2.0e-5);
    nh.param<float>("accNoise", imuAccNoise, 0.01);
    nh.param<float>("gyroNoise", imuGyrNoise, 0.001);
    nh.param<float>("gravity", imuGravity, 9.80511);

    Eigen::AngleAxisd rotX_imu(-M_PI/2, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd rotY_imu(0, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd rotZ_imu(M_PI/2, Eigen::Vector3d::UnitZ());
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

    imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for IMU message thread
    imuIntegratorOpt_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for optimization

    std::thread imu_process{processImu};
    ros::spin();

    return 0;
}
