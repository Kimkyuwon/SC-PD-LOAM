#include <fstream>
#include <math.h>
#include <vector>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <string>
#include <optional>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pclomp/ndt_omp.h>
#include <pclomp/ndt_omp_impl.hpp>
#include <pclomp/voxel_grid_covariance_omp.h>
#include <pclomp/voxel_grid_covariance_omp_impl.hpp>

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/MarkerArray.h>

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/ISAM2.h>

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include "common.h"
#include "tic_toc.h"
#include "lidarFactor.hpp"

#include "pd_loam/frame.h"
#include "pd_loam/gnss.h"

#define DEBUG_MODE_POSEGRAPH 1

using namespace gtsam;

using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)

std::string ScansDirectory, save_directory;

pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudCorner(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurf(new pcl::PointCloud<PointType>());

std::vector<Pose6D> keyframePoses;
std::vector<Pose6D> keyframePosesUpdated;
std::vector<double> keyframeTimes;
std::vector<Pose6D> keyFrameIncrements;
std::vector<pd_loam::gnss> keyframeGNSS;
int recentIdxUpdated = 0;

//for pose graph
gtsam::NonlinearFactorGraph gtSAMgraph;
bool gtSAMgraphMade = false;
bool isLoopClosed = false;
int edgeCount = 0;
gtsam::Values initialEstimate;
gtsam::ISAM2 *isam;
gtsam::Values isamCurrentEstimate;
noiseModel::Diagonal::shared_ptr priorNoise;
noiseModel::Diagonal::shared_ptr odomNoise;
noiseModel::Diagonal::shared_ptr odomINSNoise;
noiseModel::Diagonal::shared_ptr gpsNoise;
noiseModel::Base::shared_ptr robustLoopNoise;
noiseModel::Base::shared_ptr robustSeqNoise;

double keyframeMeterGap;
double keyframeDegGap, keyframeRadGap;
double translationAccumulated = 1000000.0; // large value means must add the first given frame.
double rotaionAccumulated = 1000000.0; // large value means must add the first given frame.

bool isNowKeyFrame = false;
bool poseInitialized = false;
int KeyFrameIdx = 0;
int PassIdx = 30;

Pose6D odom_pose_prev {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // init
Pose6D odom_pose_curr {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // init pose is zero
Pose6D incrementAccumulated;

double timeLaserOdometry = 0.0;
double timeLaser = 0.0;
std::queue<pd_loam::frameConstPtr> frameBuf;
std::queue<std::tuple<int, int, Eigen::Matrix4d>> LoopBuf;

pcl::KdTreeFLANN<pcl::PointXYZINormal>::Ptr kdtreeCornerLast(new pcl::KdTreeFLANN<pcl::PointXYZINormal>());
pcl::KdTreeFLANN<pcl::PointXYZINormal>::Ptr kdtreeSurfLast(new pcl::KdTreeFLANN<pcl::PointXYZINormal>());

double para_q[4] = {0, 0, 0, 1};
double para_t[3] = {0, 0, 0};

double matchingThres;

Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);
Eigen::Map<Eigen::Vector3d> t_last_curr(para_t);

std::mutex mBuf, mKF, mloop, mPG;

nav_msgs::Path PGO_path;

visualization_msgs::Marker KF_Marker, loopLine;
visualization_msgs::MarkerArray KF_Markers, loopLines;

ros::Publisher PubKeyFrameMarker, PubLoopLineMarker;
ros::Publisher PubLoopCurrent, PubLoopTarget, PubLoopLOAM;
ros::Publisher PubPGO_path;

void initNoises( void )
{
    gtsam::Vector priorNoiseVector6(6);
    priorNoiseVector6 << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12;
    priorNoise = noiseModel::Diagonal::Variances(priorNoiseVector6);

    gtsam::Vector odomNoiseVector6(6);
    odomNoiseVector6 << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4;
    odomNoise = noiseModel::Diagonal::Variances(odomNoiseVector6);

    double loopNoiseScore = 0.5; // constant is ok...
    gtsam::Vector robustNoiseVector6(6); // gtsam::Pose3 factor has 6 elements (6D)
    robustNoiseVector6 << loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore;
    robustLoopNoise = gtsam::noiseModel::Robust::Create(
                    gtsam::noiseModel::mEstimator::Cauchy::Create(1), // optional: replacing Cauchy by DCS or GemanMcClure is okay but Cauchy is empirically good.
                    gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6) );

    double loopSeqScore = 1e-8; // constant is ok...
    gtsam::Vector SeqNoiseVector6(6); // gtsam::Pose3 factor has 6 elements (6D)
    SeqNoiseVector6 << loopSeqScore, loopSeqScore, loopSeqScore, loopSeqScore, loopSeqScore, loopSeqScore;
    robustSeqNoise = gtsam::noiseModel::Robust::Create(
                    gtsam::noiseModel::mEstimator::Cauchy::Create(1), // optional: replacing Cauchy by DCS or GemanMcClure is okay but Cauchy is empirically good.
                    gtsam::noiseModel::Diagonal::Variances(SeqNoiseVector6) );
} // initNoises

Eigen::Affine3f odom2affine(nav_msgs::OdometryConstPtr odom)
{
    double x, y, z, roll, pitch, yaw;
    x = odom->pose.pose.position.x;
    y = odom->pose.pose.position.y;
    z = odom->pose.pose.position.z;
    tf::Quaternion orientation;
    tf::quaternionMsgToTF(odom->pose.pose.orientation, orientation);
    tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
    return pcl::getTransformation(x, y, z, roll, pitch, yaw);
}

gtsam::Pose3 Pose6DtoGTSAMPose3(const Pose6D& p)
{
    return gtsam::Pose3( gtsam::Rot3::RzRyRx(p.roll, p.pitch, p.yaw), gtsam::Point3(p.x, p.y, p.z) );
} // Pose6DtoGTSAMPose3


Pose6D getOdom(nav_msgs::Odometry _odom)
{
    auto tx = _odom.pose.pose.position.x;
    auto ty = _odom.pose.pose.position.y;
    auto tz = _odom.pose.pose.position.z;

    double roll, pitch, yaw;
    geometry_msgs::Quaternion quat = _odom.pose.pose.orientation;
    tf::Matrix3x3(tf::Quaternion(quat.x, quat.y, quat.z, quat.w)).getRPY(roll, pitch, yaw);

    return Pose6D{tx, ty, tz, roll, pitch, yaw};
} // getOdom

Pose6D diffTransformation(const Pose6D& _p1, const Pose6D& _p2)
{
    Eigen::Affine3f SE3_p1 = pcl::getTransformation(_p1.x, _p1.y, _p1.z, _p1.roll, _p1.pitch, _p1.yaw);
    Eigen::Affine3f SE3_p2 = pcl::getTransformation(_p2.x, _p2.y, _p2.z, _p2.roll, _p2.pitch, _p2.yaw);
    Eigen::Matrix4f SE3_delta0 = SE3_p1.matrix().inverse() * SE3_p2.matrix();
    Eigen::Affine3f SE3_delta;
    SE3_delta.matrix() = SE3_delta0;
    float dx, dy, dz, droll, dpitch, dyaw;
    pcl::getTranslationAndEulerAngles (SE3_delta, dx, dy, dz, droll, dpitch, dyaw);
    // std::cout << "delta : " << dx << ", " << dy << ", " << dz << ", " << droll << ", " << dpitch << ", " << dyaw << std::endl;

    return Pose6D{double(abs(dx)), double(abs(dy)), double(abs(dz)), double(abs(droll)), double(abs(dpitch)), double(abs(dyaw))};
} // diffTransformation

// undistort lidar point
void TransformToStart(PointType const *const pi, PointType *const po, Eigen::Quaterniond q_last_curr, Eigen::Vector3d t_last_curr)
{
    //interpolation ratio
    double s = 1.0;
    //s = 1;
    Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(s, q_last_curr);
    Eigen::Vector3d t_point_last = s * t_last_curr;
    Eigen::Vector3d point(pi->x, pi->y, pi->z);
    Eigen::Vector3d un_point = q_point_last * point + t_point_last;

    po->x = un_point.x();
    po->y = un_point.y();
    po->z = un_point.z();
    po->_PointXYZINormal::curvature = pi->_PointXYZINormal::curvature;
    po->_PointXYZINormal::normal_y = pi->_PointXYZINormal::normal_y;
    po->_PointXYZINormal::normal_z = pi->_PointXYZINormal::normal_z;
}

Eigen::Matrix4d get_TF_Matrix(const Pose6D pose)
{
    Eigen::Matrix3d rotation;
    rotation = Eigen::AngleAxisd(pose.roll, Eigen::Vector3d::UnitX())
             * Eigen::AngleAxisd(pose.pitch, Eigen::Vector3d::UnitY())
             * Eigen::AngleAxisd(pose.yaw, Eigen::Vector3d::UnitZ());
    Eigen::Quaterniond q(rotation);
    Eigen::Matrix4d TF(Eigen::Matrix4d::Identity());
    TF.block(0,0,3,3) = q.normalized().toRotationMatrix();
    TF(0,3) = pose.x;
    TF(1,3) = pose.y;
    TF(2,3) = pose.z;

    return TF;
}

void FrameHandler(const pd_loam::frameConstPtr &_frame)
{
    mBuf.lock();
    frameBuf.push(_frame);
    mBuf.unlock();
} // FrameHandler


std::optional<gtsam::Pose3> doLOAMVirtualRelative( int _loop_kf_idx, int _curr_kf_idx, Eigen::Matrix4d _diff_TF )
{
    // parse pointclouds
    pcl::PointCloud<PointType>::Ptr cureKeyframeFeatureCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr targetKeyframeFeatureCloud(new pcl::PointCloud<PointType>());
    pcl::io::loadPCDFile(ScansDirectory + std::to_string(_curr_kf_idx) + "_feature.pcd", *cureKeyframeFeatureCloud);
    pcl::io::loadPCDFile(ScansDirectory + std::to_string(_loop_kf_idx) + "_feature.pcd", *targetKeyframeFeatureCloud);

    pcl::transformPointCloud(*cureKeyframeFeatureCloud, *cureKeyframeFeatureCloud, _diff_TF);

    pcl::PointCloud<PointType>::Ptr cureKeyframeCornerCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr cureKeyframeSurfCloud(new pcl::PointCloud<PointType>());
    for (int i = 0; i < cureKeyframeFeatureCloud->points.size(); i++)
    {
        if (cureKeyframeFeatureCloud->points[i]._PointXYZINormal::normal_z == 1)
        {
            cureKeyframeCornerCloud->points.push_back(cureKeyframeFeatureCloud->points[i]);
        }
        else if(cureKeyframeFeatureCloud->points[i]._PointXYZINormal::normal_z == 2)
        {
            cureKeyframeSurfCloud->points.push_back(cureKeyframeFeatureCloud->points[i]);
        }
    }
    pcl::PointCloud<PointType>::Ptr targetKeyframeCornerCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr targetKeyframeSurfCloud(new pcl::PointCloud<PointType>());
    for (int i = 0; i < targetKeyframeFeatureCloud->points.size(); i++)
    {
        if (targetKeyframeFeatureCloud->points[i]._PointXYZINormal::normal_z == 1)
        {
            targetKeyframeCornerCloud->points.push_back(targetKeyframeFeatureCloud->points[i]);
        }
        else if(targetKeyframeFeatureCloud->points[i]._PointXYZINormal::normal_z == 2)
        {
            targetKeyframeSurfCloud->points.push_back(targetKeyframeFeatureCloud->points[i]);
        }
    }

#if DEBUG_MODE_POSEGRAPH == 1
    // loop verification
    sensor_msgs::PointCloud2 cureKeyframeCloudMsg;
    pcl::toROSMsg(*cureKeyframeSurfCloud, cureKeyframeCloudMsg);
    cureKeyframeCloudMsg.header.frame_id = "/lidar_local";
    PubLoopCurrent.publish(cureKeyframeCloudMsg);

    sensor_msgs::PointCloud2 targetKeyframeCloudMsg;
    pcl::toROSMsg(*targetKeyframeSurfCloud, targetKeyframeCloudMsg);
    targetKeyframeCloudMsg.header.frame_id = "/lidar_local";
    PubLoopTarget.publish(targetKeyframeCloudMsg);
#endif

    int cornerPointsNum = cureKeyframeCornerCloud->size();
    int surfPointsNum = cureKeyframeSurfCloud->size();

    kdtreeCornerLast->setInputCloud(targetKeyframeCornerCloud);
    kdtreeSurfLast->setInputCloud(targetKeyframeSurfCloud);

    para_t[0] = 0;
    para_t[1] = 0;
    para_t[2] = 0;
    para_q[0] = 0;
    para_q[1] = 0;
    para_q[2] = 0;
    para_q[3] = 1;
    double final_cost = 1000;
    for (size_t opti_counter = 0; opti_counter < 2; ++opti_counter)
    {
        int corner_correspondence = 0;
        int plane_correspondence = 0;

        //ceres::LossFunction *loss_function = NULL;
        ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
        ceres::LocalParameterization *q_parameterization =
            new ceres::EigenQuaternionParameterization();
        ceres::Problem::Options problem_options;

        ceres::Problem problem(problem_options);
        problem.AddParameterBlock(para_t, 3);
        problem.AddParameterBlock(para_q, 4, q_parameterization);

        pcl::PointXYZINormal pointSel;
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        std::vector<int> pointSearchInd2;
        std::vector<float> pointSearchSqDis2;
        // find correspondence for corner features
        for (int i = 0; i < cornerPointsNum; ++i)
        {
            TransformToStart(&(cureKeyframeCornerCloud->points[i]), &pointSel, q_last_curr, t_last_curr);

            kdtreeCornerLast->nearestKSearch(cureKeyframeCornerCloud->points[i], 1, pointSearchInd2, pointSearchSqDis2);
            double minPointSqDis2 = matchingThres;
            int minPointInd2 = -1;
            for (size_t s = 0; s < pointSearchInd2.size(); s++)
            {
                if (pointSearchSqDis2[s] < 6)
                {
                    double pointSqDis = (targetKeyframeCornerCloud->points[pointSearchInd2[s]].x - cureKeyframeCornerCloud->points[i].x) *
                                        (targetKeyframeCornerCloud->points[pointSearchInd2[s]].x - cureKeyframeCornerCloud->points[i].x) +
                                        (targetKeyframeCornerCloud->points[pointSearchInd2[s]].y - cureKeyframeCornerCloud->points[i].y) *
                                        (targetKeyframeCornerCloud->points[pointSearchInd2[s]].y - cureKeyframeCornerCloud->points[i].y) +
                                        (targetKeyframeCornerCloud->points[pointSearchInd2[s]].z - cureKeyframeCornerCloud->points[i].z) *
                                        (targetKeyframeCornerCloud->points[pointSearchInd2[s]].z - cureKeyframeCornerCloud->points[i].z);

                    if (pointSqDis < minPointSqDis2)
                    {
                        // find nearer point
                        minPointSqDis2 = pointSqDis;
                        minPointInd2 = pointSearchInd2[s];
                    }
                }
            }
            if (minPointInd2 >= 0)
            {
                pcl::PointCloud<pcl::PointXYZINormal> clusters;
                for (size_t j = 0; j < targetKeyframeCornerCloud->points.size(); j++)
                {
                    if (targetKeyframeCornerCloud->points[minPointInd2]._PointXYZINormal::normal_y != targetKeyframeCornerCloud->points[j]._PointXYZINormal::normal_y)
                    {
                        continue;
                    }
                    clusters.push_back(targetKeyframeCornerCloud->points[j]);
                }
                if (clusters.points.size() >= 5)
                {
                    Eigen::Matrix3d covariance = getCovariance(clusters);
                    Eigen::Vector3d mean = getMean(clusters);
                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covariance);

                    // if is indeed line feature
                    // note Eigen library sort eigenvalues in increasing order
                    Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);

                    Eigen::Vector3d curr_point(cureKeyframeCornerCloud->points[i].x,
                                               cureKeyframeCornerCloud->points[i].y,
                                               cureKeyframeCornerCloud->points[i].z);

                    Eigen::Vector3d point_on_line;
                    point_on_line(0) = mean(0);
                    point_on_line(1) = mean(1);
                    point_on_line(2) = mean(2);
                    Eigen::Vector3d point_a, point_b;
                    point_a = unit_direction + point_on_line;
                    point_b = -unit_direction + point_on_line;

                    ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, point_a, point_b, 1.0);
                    problem.AddResidualBlock(cost_function, loss_function, para_t, para_q);
                    corner_correspondence++;

                }
            }
        }

        // find correspondence for plane features
        for (int i = 0; i < surfPointsNum; ++i)
        {
            TransformToStart(&(cureKeyframeSurfCloud->points[i]), &pointSel, q_last_curr, t_last_curr);
            kdtreeSurfLast->nearestKSearch(cureKeyframeSurfCloud->points[i], 1, pointSearchInd2, pointSearchSqDis2);

            double minPointSqDis2 = matchingThres;
            int minPointInd2 = -1;
            for (size_t s = 0; s < pointSearchInd2.size(); s++)
            {
                if (pointSearchSqDis2[s] < 6)
                {
                    double pointSqDis = (targetKeyframeSurfCloud->points[pointSearchInd2[s]].x - cureKeyframeSurfCloud->points[i].x) *
                                        (targetKeyframeSurfCloud->points[pointSearchInd2[s]].x - cureKeyframeSurfCloud->points[i].x) +
                                        (targetKeyframeSurfCloud->points[pointSearchInd2[s]].y - cureKeyframeSurfCloud->points[i].y) *
                                        (targetKeyframeSurfCloud->points[pointSearchInd2[s]].y - cureKeyframeSurfCloud->points[i].y) +
                                        (targetKeyframeSurfCloud->points[pointSearchInd2[s]].z - cureKeyframeSurfCloud->points[i].z) *
                                        (targetKeyframeSurfCloud->points[pointSearchInd2[s]].z - cureKeyframeSurfCloud->points[i].z);

                    if (pointSqDis < minPointSqDis2)
                    {
                        // find nearer point
                        minPointSqDis2 = pointSqDis;
                        minPointInd2 = pointSearchInd2[s];
                    }
                }
            }
            if (minPointInd2 >= 0)
            {
                pcl::PointCloud<pcl::PointXYZINormal> clusters;
                for (size_t j = 0; j < targetKeyframeSurfCloud->points.size(); j++)
                {
                    if (targetKeyframeSurfCloud->points[minPointInd2]._PointXYZINormal::normal_y != targetKeyframeSurfCloud->points[j]._PointXYZINormal::normal_y)
                    {
                        continue;
                    }
                    clusters.push_back(targetKeyframeSurfCloud->points[j]);
                }

                if (clusters.points.size() >= 5)
                {
                    Eigen::Matrix3d covariance = getCovariance(clusters);
                    Eigen::Vector3d mean = getMean(clusters);
                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covariance);

                    // if is indeed plane feature
                    // note Eigen library sort eigenvalues in increasing order
                    Eigen::Vector3d direction1 = saes.eigenvectors().col(2);
                    Eigen::Vector3d direction2 = saes.eigenvectors().col(1);

                    Eigen::Vector3d curr_point(cureKeyframeSurfCloud->points[i].x,
                                               cureKeyframeSurfCloud->points[i].y,
                                               cureKeyframeSurfCloud->points[i].z);

                    Eigen::Vector3d point_on_surf;
                    point_on_surf = mean;
                    Eigen::Vector3d point_a, point_b, point_c;
                    point_a = direction1 + point_on_surf;
                    point_b = -direction1 + point_on_surf;
                    point_c = direction2 + point_on_surf;

                    ceres::CostFunction *cost_function = LidarPlaneFactor::Create(curr_point, point_a, point_b, point_c, 1.0);
                    problem.AddResidualBlock(cost_function, loss_function, para_t, para_q);
                    plane_correspondence++;
                }
            }
        }
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.max_num_iterations = 4;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        final_cost = summary.final_cost/(corner_correspondence + plane_correspondence);
        //std::cout<<"cost : "<<final_cost<<std::endl;
    }

    if (final_cost < 0.02)
    {
        Eigen::Matrix3d rot = q_last_curr.toRotationMatrix();
        Eigen::Matrix4d TF (Eigen::Matrix4d::Identity());
        TF.block(0,0,3,3) = rot;
        TF(0,3) = t_last_curr(0);   TF(1,3) = t_last_curr(1);   TF(2,3) = t_last_curr(2);

        for (int i = 0; i < int(cureKeyframeSurfCloud->size()); ++i)
        {
            const auto &pointFrom = cureKeyframeSurfCloud->points[i];
            PointType tmp;
            tmp.x = rot(0,0) * pointFrom.x + rot(0,1) * pointFrom.y + rot(0,2) * pointFrom.z + t_last_curr(0);
            tmp.y = rot(1,0) * pointFrom.x + rot(1,1) * pointFrom.y + rot(1,2) * pointFrom.z + t_last_curr(1);
            tmp.z = rot(2,0) * pointFrom.x + rot(2,1) * pointFrom.y + rot(2,2) * pointFrom.z + t_last_curr(2);
            tmp.intensity = pointFrom.intensity;
            tmp._PointXYZINormal::curvature = pointFrom._PointXYZINormal::curvature;
            tmp._PointXYZINormal::normal_y = pointFrom._PointXYZINormal::normal_y;
            tmp._PointXYZINormal::normal_z = pointFrom._PointXYZINormal::normal_z;
            cureKeyframeSurfCloud->points[i] = tmp;
        }

#if DEBUG_MODE_POSEGRAPH == 1
        sensor_msgs::PointCloud2 LOAMCloudMsg;
        pcl::toROSMsg(*cureKeyframeSurfCloud, LOAMCloudMsg);
        LOAMCloudMsg.header.frame_id = "/lidar_local";
        PubLoopLOAM.publish(LOAMCloudMsg);
#endif

        Eigen::Matrix4d final_TF = _diff_TF * TF;
        Eigen::Matrix3d final_rot = final_TF.block(0,0,3,3);
        Eigen::Quaterniond final_q(final_rot);
        // Get pose transformation
        double roll, pitch, yaw;
        tf::Matrix3x3(tf::Quaternion(final_q.x(), final_q.y(), final_q.z(), final_q.w())).getRPY(roll, pitch, yaw);
        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(final_TF(0,3), final_TF(1,3), final_TF(2,3)));
        gtsam::Pose3 poseTo = Pose3(Rot3::RzRyRx(0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0));
        std::cout<<"x : "<<poseFrom.x()<<", y : "<<poseFrom.y()<<", z : "<<poseFrom.z()<<", r : "<<rad2deg(roll)<<", p : "<<rad2deg(pitch)<<", yaw : "<<rad2deg(yaw)<<std::endl;

        return poseFrom.between(poseTo);
    }
    else
    {
        return std::nullopt;
    }
}

void updatePoses(void)
{
    mKF.lock();
    PGO_path.poses.clear();
    for (int node_idx=0; node_idx < int(isamCurrentEstimate.size()); node_idx++)
    {
        Pose6D& p =keyframePosesUpdated[node_idx];
        p.x = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().x();
        p.y = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().y();
        p.z = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().z();
        p.roll = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().roll();
        p.pitch = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().pitch();
        p.yaw = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().yaw();
        keyframePosesUpdated[node_idx] = p;
        geometry_msgs::PoseStamped poseStampPGO;
        poseStampPGO.header.frame_id = "/body";
        poseStampPGO.pose.position.x = p.x;
        poseStampPGO.pose.position.y = p.y;
        poseStampPGO.pose.position.z = p.z;
        poseStampPGO.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(p.roll, p.pitch, p.yaw);

        PGO_path.header.frame_id = "/body";
        PGO_path.poses.push_back(poseStampPGO);
    }
    PGO_path.header.stamp = ros::Time().fromSec(keyframeTimes.back());
    PGO_path.header.frame_id = "/body";
    PubPGO_path.publish(PGO_path);
    isLoopClosed = false;
    mKF.unlock();
} // updatePoses

void runISAM2opt(void)
{
    // called when a variable added
    isam->update(gtSAMgraph, initialEstimate);
    isam->update();
    if (isLoopClosed == true)
    {
        isam->update();
        isam->update();
        isam->update();
        isam->update();
    }
    gtSAMgraph.resize(0);
    initialEstimate.clear();

    isamCurrentEstimate = isam->calculateEstimate();
    recentIdxUpdated = int(isamCurrentEstimate.size());
    updatePoses();
}

void process_pg()
{
    while(1)
    {
        while(!frameBuf.empty())
        {
            mBuf.lock();
            if(frameBuf.empty())
            {
                mBuf.unlock();
                break;
            }

            // Time equal check
            timeLaserOdometry = frameBuf.front()->pose.header.stamp.toSec();
            timeLaser = frameBuf.front()->fullPC.header.stamp.toSec();
            if (timeLaserOdometry < timeLaser)
            {
                frameBuf.pop();
                ROS_WARN("Time is not synced.");
            }

            TicToc t_pg;

            laserCloudFullRes->clear();
            laserCloudCorner->clear();
            laserCloudSurf->clear();
            pcl::fromROSMsg(frameBuf.front()->fullPC, *laserCloudFullRes);
            pcl::fromROSMsg(frameBuf.front()->CornerPC, *laserCloudCorner);
            pcl::fromROSMsg(frameBuf.front()->SurfPC, *laserCloudSurf);
            Pose6D pose_curr = getOdom(frameBuf.front()->pose);
            pd_loam::gnss gnss_curr = frameBuf.front()->GNSS;
            int frameIdx = frameBuf.front()->frame_idx;
            frameBuf.pop();
            odom_pose_prev = odom_pose_curr;
            odom_pose_curr = pose_curr;
            Pose6D dtf = diffTransformation(odom_pose_prev, odom_pose_curr); // dtf means delta_transform

            double delta_translation = sqrt(dtf.x*dtf.x + dtf.y*dtf.y + dtf.z*dtf.z); // note: absolute value.
            translationAccumulated += delta_translation;
            rotaionAccumulated += (dtf.roll + dtf.pitch + dtf.yaw); // sum just naive approach.

            incrementAccumulated.x += dtf.x;   incrementAccumulated.y += dtf.y;   incrementAccumulated.z += dtf.z;
            incrementAccumulated.roll += dtf.roll; incrementAccumulated.pitch += dtf.pitch;   incrementAccumulated.yaw += dtf.yaw;
            if(translationAccumulated > keyframeMeterGap || rotaionAccumulated > keyframeRadGap || KeyFrameIdx == 0)
            {
                isNowKeyFrame = true;
                KF_Marker.header.stamp = ros::Time().fromSec(timeLaserOdometry);
                KF_Marker.action = visualization_msgs::Marker::ADD;
                KF_Marker.pose.position.x = pose_curr.x;
                KF_Marker.pose.position.y = pose_curr.y;
                KF_Marker.pose.position.z = pose_curr.z;
                KF_Marker.id = frameIdx;
                KF_Markers.markers.push_back(KF_Marker);
                PubKeyFrameMarker.publish(KF_Markers);
                translationAccumulated = 0.0; // reset
                rotaionAccumulated = 0.0; // reset
                KeyFrameIdx++;
            }
            else
            {
                isNowKeyFrame = false;
                //printf("pose graph not key frame time %f ms ++++++++++\n", t_pg.toc());
            }
            mBuf.unlock();

            if(!isNowKeyFrame)   continue;

            mKF.lock();

            pcl::PointCloud<PointType>::Ptr laserCloudFeature (new pcl::PointCloud<PointType>());
            *laserCloudFeature += *laserCloudCorner;
            *laserCloudFeature += *laserCloudSurf;

            pcl::io::savePCDFileBinary(ScansDirectory + std::to_string(keyframePoses.size()) + "_full.pcd", *laserCloudFullRes);
            pcl::io::savePCDFileBinary(ScansDirectory + std::to_string(keyframePoses.size()) + "_feature.pcd", *laserCloudFeature);
            keyframePoses.push_back(pose_curr);
            keyframePosesUpdated.push_back(pose_curr);
            keyFrameIncrements.push_back(incrementAccumulated);
            keyframeTimes.push_back(timeLaserOdometry);
            keyframeGNSS.push_back(gnss_curr);

            incrementAccumulated.x = 0; incrementAccumulated.y = 0; incrementAccumulated.z = 0;
            incrementAccumulated.roll = 0;  incrementAccumulated.pitch = 0; incrementAccumulated.yaw = 0;

            if (keyframePoses.size() > PassIdx)
            {
                Pose6D To_pose = keyframePoses.back();
                Eigen::Matrix4d to_TF = get_TF_Matrix(To_pose);

                std::vector<float> LOOP_DIST;
                std::vector<int> LOOP_IDX;
                std::vector<Eigen::Matrix4d> FROM_TFs;
                for (size_t i = 0; i < keyframePoses.size() - PassIdx; i++)
                {
                    Pose6D From_pose = keyframePoses[i];
                    Eigen::Matrix4d from_TF = get_TF_Matrix(From_pose);

                    float loop_dist = sqrt(pow((to_TF(0,3)-from_TF(0,3)),2)+pow((to_TF(1,3)-from_TF(1,3)),2)+pow((to_TF(2,3)-from_TF(2,3)),2));

                    if (loop_dist > 2)  continue;
                    LOOP_DIST.push_back(loop_dist);
                    LOOP_IDX.push_back(i);
                    FROM_TFs.push_back(from_TF);
                }
                int min_idx = -1;
                if (LOOP_DIST.size() > 0)
                {
                    float min_loop_dist = LOOP_DIST.front();
                    for (size_t k = 1; k < LOOP_DIST.size(); k++)
                    {
                        Eigen::Matrix4d temp_TF = FROM_TFs[k].inverse() * to_TF;
                        Eigen::Affine3f SE3_delta;
                        SE3_delta.matrix() = temp_TF.cast<float>();
                        float dx, dy, dz, droll, dpitch, dyaw;
                        pcl::getTranslationAndEulerAngles (SE3_delta, dx, dy, dz, droll, dpitch, dyaw);
                        if (min_loop_dist > LOOP_DIST[k] /*&& fabs(rad2deg(dyaw)) < 60*/)
                        {
                            min_loop_dist = LOOP_DIST[k];
                            min_idx = k;
                        }
                    }
                    if (min_idx > -1)
                    {
                        ROS_INFO("loop detected!!");
                        loopLine.header.stamp = ros::Time().fromSec(timeLaserOdometry);
                        geometry_msgs::Point p;
                        p.x = FROM_TFs[min_idx](0,3);    p.y = FROM_TFs[min_idx](1,3);    p.z = FROM_TFs[min_idx](2,3);
                        loopLine.points.push_back(p);
                        p.x = to_TF(0,3);    p.y = to_TF(1,3);    p.z = to_TF(2,3);
                        loopLine.points.push_back(p);
                        PubLoopLineMarker.publish(loopLine);

                        Eigen::Matrix4d delta_TF = FROM_TFs[min_idx].inverse() * to_TF;

                        LoopBuf.push(std::tuple<int, int, Eigen::Matrix4d>(LOOP_IDX[min_idx], keyframePoses.size()-1, delta_TF));
                    }
                }
//                for (size_t i = 2; i < 5; i++)
//                {
//                    Pose6D From_pose = keyframePoses[keyframePoses.size()-i];
//                    Eigen::Matrix4d from_TF = get_TF_Matrix(From_pose);
//                    Eigen::Matrix4d delta_TF = from_TF.inverse() * to_TF;

//                    LoopBuf.push(std::tuple<int, int, Eigen::Matrix4d>(keyframePoses.size()-i, keyframePoses.size()-1, delta_TF));
//                }
            }
            mKF.unlock();

            mPG.lock();
            const int prev_node_idx = keyframePoses.size() - 2;
            const int curr_node_idx = keyframePoses.size() - 1; // becuase cpp starts with 0 (actually this index could be any number, but for simple implementation, we follow sequential indexing)
            if(!gtSAMgraphMade /* prior node */)
            {
                const int init_node_idx = 0;
                gtsam::Pose3 poseOrigin = Pose6DtoGTSAMPose3(keyframePoses.at(init_node_idx));

                // prior factor
                gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(init_node_idx, poseOrigin, priorNoise));
                initialEstimate.insert(init_node_idx, poseOrigin);

                gtSAMgraphMade = true;
            }
            else /* consecutive node (and odom factor) after the prior added */
            {
                gtsam::Pose3 poseFrom = Pose6DtoGTSAMPose3(keyframePoses.at(prev_node_idx));
                gtsam::Pose3 poseTo = Pose6DtoGTSAMPose3(keyframePoses.at(curr_node_idx));

                // odom factor
                gtsam::Pose3 relPose = poseFrom.between(poseTo);
                gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(prev_node_idx, curr_node_idx, relPose, odomNoise));

                //gnss factor
                if (keyframeGNSS.back().eastPos_std < 0.1 && keyframeGNSS.back().northPos_std < 0.1)
                {
                    float up_std;
                    double upPos;
                    if (keyframeGNSS.back().upPos_std < 0.2)
                    {
                        up_std = keyframeGNSS.back().upPos_std;
                        upPos = keyframeGNSS.back().upPos;
                    }
                    else
                    {
                        up_std = 0.1;
                        upPos = poseTo.z();
                    }
                    gpsNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(3) << pow(keyframeGNSS.back().eastPos_std,2), pow(keyframeGNSS.back().northPos_std,2), pow(up_std,2)).finished()); // e,n,u
                    gtsam::GPSFactor gps_factor(curr_node_idx, gtsam::Point3(keyframeGNSS.back().eastPos, keyframeGNSS.back().northPos, upPos), gpsNoise);
                    gtSAMgraph.add(gps_factor);
                }
                initialEstimate.insert(curr_node_idx, poseTo);
            }
            mPG.unlock();
        }
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

void process_edge(void)
{
    while(1)
    {
        while (!LoopBuf.empty())
        {

            mloop.lock();
            std::tuple<int, int, Eigen::Matrix4d> loop_idx_pair = LoopBuf.front();
            LoopBuf.pop();
            const int prev_node_idx = get<0>(loop_idx_pair);
            const int curr_node_idx = get<1>(loop_idx_pair);
            const Eigen::Matrix4d diff_TF = get<2>(loop_idx_pair);

            TicToc t_loam;
            auto relative_pose_optional = doLOAMVirtualRelative(prev_node_idx, curr_node_idx, diff_TF);
            printf("loam matching time %f ms ++++++++++\n", t_loam.toc());
            if(relative_pose_optional)
            {
                gtsam::Pose3 relative_pose = relative_pose_optional.value();
                if ((curr_node_idx - prev_node_idx) > 3)
                {
                    loopLine.header.stamp = ros::Time().fromSec(timeLaserOdometry);
                    geometry_msgs::Point p;
                    p.x = keyframePoses[prev_node_idx].x;    p.y = keyframePoses[prev_node_idx].y;    p.z = keyframePoses[prev_node_idx].z;
                    loopLine.points.push_back(p);
                    p.x = keyframePoses[curr_node_idx].x;    p.y = keyframePoses[curr_node_idx].y;    p.z = keyframePoses[curr_node_idx].z;
                    loopLine.points.push_back(p);
                    PubLoopLineMarker.publish(loopLine);
                    gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(prev_node_idx, curr_node_idx, relative_pose, robustLoopNoise));
                }
                else
                {
                    gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(prev_node_idx, curr_node_idx, relative_pose, robustSeqNoise));
                    isLoopClosed = true;
                }
            }
            mloop.unlock();
        }

        // wait (must required for running the while loop)
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
} // process_edge

void process_optimization(void)
{
    float hz = 1;
    ros::Rate rate(hz);
    while (ros::ok())
    {
        rate.sleep();
        if(isNowKeyFrame == true)
        {
            TicToc t_opt;
            mPG.lock();
            runISAM2opt();
            mPG.unlock();
            printf("pose graph optimization time %f ms ++++++++++\n", t_opt.toc());
        }
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "laserPGO");
    ros::NodeHandle nh;

    // save directories
    nh.param<std::string>("save_directory", save_directory, "/"); // pose assignment every k m move

    ScansDirectory = save_directory + "Scans/";
    auto unused = system((std::string("exec rm -r ") + ScansDirectory).c_str());
    unused = system((std::string("mkdir -p ") + ScansDirectory).c_str());

    nh.param<double>("keyframe_meter_gap", keyframeMeterGap, 2.0); // pose assignment every k m move
    nh.param<double>("keyframe_deg_gap", keyframeDegGap, 10.0); // pose assignment every k deg rot
    keyframeRadGap = deg2rad(keyframeDegGap);
    ROS_INFO("KF gap : %f, %f", keyframeMeterGap, keyframeDegGap);

    nh.param<double>("matching_threshold",matchingThres, 4);

    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    isam = new ISAM2(parameters);
    initNoises();

    KF_Marker.type = visualization_msgs::Marker::SPHERE;
    KF_Marker.header.frame_id = "/body";
    KF_Marker.color.g = 1.0;
    KF_Marker.color.a = 0.7;
    KF_Marker.scale.x = 0.2;
    KF_Marker.scale.y = 0.2;
    KF_Marker.scale.z = 0.2;

    loopLine.type = visualization_msgs::Marker::LINE_LIST;
    loopLine.action = visualization_msgs::Marker::ADD;
    loopLine.color.b = 1.0; loopLine.color.a = 0.7;
    loopLine.scale.x = 0.05;
    loopLine.header.frame_id = "/body";

    ros::Subscriber subFrame = nh.subscribe<pd_loam::frame>("/mapping/frame", 100, FrameHandler);

    PubPGO_path = nh.advertise<nav_msgs::Path>("/posegraph/PGO_path", 100);

    PubKeyFrameMarker = nh.advertise<visualization_msgs::MarkerArray>("/posegraph/KF",100);

    PubLoopLineMarker = nh.advertise<visualization_msgs::Marker>("/posegraph/loopLine", 100);
#if DEBUG_MODE_POSEGRAPH == 1
    PubLoopCurrent = nh.advertise<sensor_msgs::PointCloud2>("/loop_current", 100);
    PubLoopTarget = nh.advertise<sensor_msgs::PointCloud2>("/loop_target", 100);
    PubLoopLOAM = nh.advertise<sensor_msgs::PointCloud2>("/loop_loam", 100);
#endif

    std::thread posegraph {process_pg}; // pose graph construction
    std::thread edge_calculation {process_edge};    //NDT based edge measurement calculation
    std::thread graph_optimization {process_optimization};
    ros::spin();

    return 0;
}
