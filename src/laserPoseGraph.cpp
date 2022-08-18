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
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>

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

#include <eigen3/Eigen/Dense>

#include <ceres/ceres.h>

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

#include "common.h"
#include "tic_toc.h"
#include "lidarFactor.hpp"

#include "scancontext/Scancontext.h"
#include "pd_loam/frame.h"

#define DISTORTION 0
#define DEBUG_MODE_POSEGRAPH 0

using namespace gtsam;

using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)

double keyframeMeterGap;
double keyframeDegGap, keyframeRadGap;
double translationAccumulated = 1000000.0; // large value means must add the first given frame.
double rotaionAccumulated = 1000000.0; // large value means must add the first given frame.

Pose6D incrementAccumulated;

bool isNowKeyFrame = false;
bool poseInitialized = false;
bool gpsInitialized = false;
bool getGT = false;
int KeyFrameIdx = 0;

Pose6D odom_pose_prev {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // init
Pose6D odom_pose_curr {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // init pose is zero
std::string save_directory;
std::string ScansDirectory;

pcl::VoxelGrid<PointType> downSizeFilterScancontext, downSizeFilterICP;
SCManager scManager;
double scDistThres, scMaximumRadius, scResolution;

pcl::KdTreeFLANN<pcl::PointXYZINormal>::Ptr kdtreeCornerLast(new pcl::KdTreeFLANN<pcl::PointXYZINormal>());
pcl::KdTreeFLANN<pcl::PointXYZINormal>::Ptr kdtreeSurfLast(new pcl::KdTreeFLANN<pcl::PointXYZINormal>());

Eigen::Vector3d OriginXYZ;
Eigen::Vector3d gpsENU;

// q_curr_last(x, y, z, w), t_curr_last
double para_q[4] = {0, 0, 0, 1};
double para_t[3] = {0, 0, 0};

double matchingThres;

Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);
Eigen::Map<Eigen::Vector3d> t_last_curr(para_t);

std::queue<pd_loam::frameConstPtr> frameBuf;
std::queue<std::tuple<int, int, float>> scLoopBuf;
std::queue<nav_msgs::OdometryConstPtr> imuOdomBuf;
std::queue<nav_msgs::OdometryConstPtr> lidarOdomBuf;
std::queue<std::pair<Eigen::Vector3d, Eigen::Vector3d>> gpsBuf;

std::mutex mBuf;
std::mutex mKF;
std::mutex mPG;
std::mutex mloop;
std::mutex mRecentPose;

double timeLaserOdometry = 0.0;
double timeLaser = 0.0;

pcl::PointCloud<PointType>::Ptr laserCloudMapPGO(new pcl::PointCloud<PointType>());
pcl::VoxelGrid<PointType> downSizeFilterMapPGO;

pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudCorner(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurf(new pcl::PointCloud<PointType>());

std::vector<Pose6D> keyframePoses;
std::vector<Pose6D> keyframePosesUpdated;
std::vector<double> keyframeTimes;
std::vector<Pose6D> keyFrameIncrements;
int recentIdxUpdated = 0;

int Prev_prev_node_idx = 0;
int Prev_curr_node_idx = 0;

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
noiseModel::Diagonal::shared_ptr odomImuNoise;
noiseModel::Diagonal::shared_ptr gpsNoise;
noiseModel::Base::shared_ptr robustLoopNoise;

nav_msgs::Path gps_path;
nav_msgs::Path gt_path;

std::vector<std::string> edges_str; // used in writeEdge

visualization_msgs::Marker KF_Marker, loopLine;
visualization_msgs::MarkerArray KF_Markers, loopLines;

ros::Publisher PubKeyFrameMarker, PubLoopLineMarker;
ros::Publisher PubLoopCurrent, PubLoopTarget, PubLoopLOAM, PubLoopTarget2;
ros::Publisher PubOdomAftPGO, PubPathAftPGO, PubPathPose, PubPathGT, PubPathGPS;

void initNoises( void )
{
    gtsam::Vector priorNoiseVector6(6);
    priorNoiseVector6 << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6;
    priorNoise = noiseModel::Diagonal::Variances(priorNoiseVector6);

    gtsam::Vector odomNoiseVector6(6);
    odomNoiseVector6 << 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4;
    odomNoise = noiseModel::Diagonal::Variances(odomNoiseVector6);
    odomNoiseVector6 << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4;
    odomImuNoise = noiseModel::Diagonal::Variances(odomNoiseVector6);

    double loopNoiseScore = 1; // constant is ok...
    gtsam::Vector robustNoiseVector6(6); // gtsam::Pose3 factor has 6 elements (6D)
    robustNoiseVector6 << loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore;
    robustLoopNoise = gtsam::noiseModel::Robust::Create(
                    gtsam::noiseModel::mEstimator::Cauchy::Create(1), // optional: replacing Cauchy by DCS or GemanMcClure is okay but Cauchy is empirically good.
                    gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6) );
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

std::string getVertexStr(const int _node_idx, const gtsam::Pose3& _Pose)
{
    gtsam::Point3 t = _Pose.translation();
    gtsam::Rot3 R = _Pose.rotation();

    std::string curVertexInfo {
        "VERTEX_SE3:QUAT " + std::to_string(_node_idx) + " "
        + std::to_string(t.x()) + " " + std::to_string(t.y()) + " " + std::to_string(t.z())  + " "
        + std::to_string(R.toQuaternion().x()) + " " + std::to_string(R.toQuaternion().y()) + " "
        + std::to_string(R.toQuaternion().z()) + " " + std::to_string(R.toQuaternion().w()) };

    // pgVertexSaveStream << curVertexInfo << std::endl;
    // vertices_str.emplace_back(curVertexInfo);
    return curVertexInfo;
}

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
    Eigen::Affine3f SE3_delta; SE3_delta.matrix() = SE3_delta0;
    float dx, dy, dz, droll, dpitch, dyaw;
    pcl::getTranslationAndEulerAngles (SE3_delta, dx, dy, dz, droll, dpitch, dyaw);
    // std::cout << "delta : " << dx << ", " << dy << ", " << dz << ", " << droll << ", " << dpitch << ", " << dyaw << std::endl;

    return Pose6D{double(abs(dx)), double(abs(dy)), double(abs(dz)), double(abs(droll)), double(abs(dpitch)), double(abs(dyaw))};
} // diffTransformation

pcl::PointCloud<PointType>::Ptr local2global(const pcl::PointCloud<PointType>::Ptr &cloudIn, const Pose6D& tf)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur = pcl::getTransformation(tf.x, tf.y, tf.z, tf.roll, tf.pitch, tf.yaw);

    for (int i = 0; i < cloudSize; ++i)
    {
        const auto &pointFrom = cloudIn->points[i];
        cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
        cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
        cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
        cloudOut->points[i].intensity = pointFrom.intensity;
    }

    return cloudOut;
}

void saveSCD(std::string fileName, Eigen::MatrixXd matrix, std::string delimiter = " ")
{
    // delimiter: ", " or " " etc.

    int precision = 3; // or Eigen::FullPrecision, but SCD does not require such accruate precisions so 3 is enough.
    const static Eigen::IOFormat the_format(precision, Eigen::DontAlignCols, delimiter, "\n");

    std::ofstream file(fileName);
    if (file.is_open())
    {
        file << matrix.format(the_format);
        file.close();
    }
}

gtsam::Pose3 Pose6DtoGTSAMPose3(const Pose6D& p)
{
    return gtsam::Pose3( gtsam::Rot3::RzRyRx(p.roll, p.pitch, p.yaw), gtsam::Point3(p.x, p.y, p.z) );
} // Pose6DtoGTSAMPose3

void pubPath( void )
{
    // pub odom and path
    nav_msgs::Odometry odomAftPGO;
    nav_msgs::Path pathAftPGO, pathPose, pathGT;
    pathAftPGO.header.frame_id = "/body";
    mKF.lock();
    // for (int node_idx=0; node_idx < int(keyframePosesUpdated.size()) - 1; node_idx++) // -1 is just delayed visualization (because sometimes mutexed while adding(push_back) a new one)
    for (int node_idx=0; node_idx < recentIdxUpdated; node_idx++) // -1 is just delayed visualization (because sometimes mutexed while adding(push_back) a new one)
    {
        const Pose6D& pose_est = keyframePosesUpdated.at(node_idx);
        const Pose6D& pose = keyframePoses.at(node_idx);

        nav_msgs::Odometry odomAftPGOthis;
        odomAftPGOthis.header.frame_id = "/body";
        odomAftPGOthis.child_frame_id = "/aft_pgo";
        odomAftPGOthis.header.stamp = ros::Time().fromSec(keyframeTimes.at(node_idx));
        odomAftPGOthis.pose.pose.position.x = pose_est.x;
        odomAftPGOthis.pose.pose.position.y = pose_est.y;
        odomAftPGOthis.pose.pose.position.z = pose_est.z;
        odomAftPGOthis.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(pose_est.roll, pose_est.pitch, pose_est.yaw);
        odomAftPGO = odomAftPGOthis;

        nav_msgs::Odometry odomPose;
        odomPose.header.frame_id = "/body";
        odomPose.header.stamp = ros::Time().fromSec(keyframeTimes.at(node_idx));
        odomPose.pose.pose.position.x = pose.x;
        odomPose.pose.pose.position.y = pose.y;
        odomPose.pose.pose.position.z = pose.z;
        odomPose.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(pose.roll, pose.pitch, pose.yaw);


        if (getGT == true)
        {
            double minTime = 100000;
            size_t gtIdx = 0;
            for (size_t i = 0; i < gt_path.poses.size(); i++)
            {
                if (fabs(keyframeTimes.at(node_idx) - gt_path.poses[i].header.stamp.toSec()) < minTime)
                {
                    minTime = fabs(keyframeTimes.at(node_idx) - gt_path.poses[i].header.stamp.toSec());
                    gtIdx = i;
                }
                if (fabs(keyframeTimes.at(node_idx) - gt_path.poses[i].header.stamp.toSec()) < 0.05) break;
            }

            nav_msgs::Odometry odomGT_Pose;
            odomGT_Pose.header.stamp = gt_path.poses[gtIdx].header.stamp;
            odomGT_Pose.pose.pose.position.x = gt_path.poses[gtIdx].pose.position.x;
            odomGT_Pose.pose.pose.position.y = gt_path.poses[gtIdx].pose.position.y;
            odomGT_Pose.pose.pose.position.z = gt_path.poses[gtIdx].pose.position.z;
            odomGT_Pose.pose.pose.orientation.w = gt_path.poses[gtIdx].pose.orientation.w;
            odomGT_Pose.pose.pose.orientation.x = gt_path.poses[gtIdx].pose.orientation.x;
            odomGT_Pose.pose.pose.orientation.y = gt_path.poses[gtIdx].pose.orientation.y;
            odomGT_Pose.pose.pose.orientation.z = gt_path.poses[gtIdx].pose.orientation.z;

            geometry_msgs::PoseStamped poseStampGT_Pose;
            poseStampGT_Pose.header = odomGT_Pose.header;
            poseStampGT_Pose.pose = odomGT_Pose.pose.pose;

            pathGT.header.stamp = odomGT_Pose.header.stamp;
            pathGT.header.frame_id = "/body";
            pathGT.poses.push_back(poseStampGT_Pose);

            odomPose.pose.pose.position.z = gt_path.poses[gtIdx].pose.position.z;
            odomAftPGOthis.pose.pose.position.z = gt_path.poses[gtIdx].pose.position.z;
        }
        geometry_msgs::PoseStamped poseStampAftPGO;
        poseStampAftPGO.header = odomAftPGOthis.header;
        poseStampAftPGO.pose = odomAftPGOthis.pose.pose;

        pathAftPGO.header.stamp = odomAftPGOthis.header.stamp;
        pathAftPGO.header.frame_id = "/body";
        pathAftPGO.poses.push_back(poseStampAftPGO);

        geometry_msgs::PoseStamped poseStampPose;
        poseStampPose.header = odomPose.header;
        poseStampPose.pose = odomPose.pose.pose;

        pathPose.header.stamp = odomPose.header.stamp;
        pathPose.header.frame_id = "/body";
        pathPose.poses.push_back(poseStampPose);
    }
    mKF.unlock();
    PubOdomAftPGO.publish(odomAftPGO); // last pose
    PubPathAftPGO.publish(pathAftPGO); // poses
    PubPathPose.publish(pathPose);
    PubPathGT.publish(pathGT);

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(odomAftPGO.pose.pose.position.x, odomAftPGO.pose.pose.position.y, odomAftPGO.pose.pose.position.z));
    q.setW(odomAftPGO.pose.pose.orientation.w);
    q.setX(odomAftPGO.pose.pose.orientation.x);
    q.setY(odomAftPGO.pose.pose.orientation.y);
    q.setZ(odomAftPGO.pose.pose.orientation.z);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, odomAftPGO.header.stamp, "/body", "/aft_pgo"));
} // pubPath

void updatePoses(void)
{
    mKF.lock();
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
    }
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

// undistort lidar point
void TransformToStart(PointType const *const pi, PointType *const po, Eigen::Quaterniond q_last_curr, Eigen::Vector3d t_last_curr)
{
    //interpolation ratio
    double s;
    if (DISTORTION)
        s = (pi->_PointXYZINormal::curvature - int(pi->_PointXYZINormal::curvature)) / 0.1;
    else
        s = 1.0;
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

std::optional<gtsam::Pose3> doLOAMVirtualRelative( int _loop_kf_idx, int _curr_kf_idx, float _diff_yaw )
{
    // parse pointclouds
    pcl::PointCloud<PointType>::Ptr cureKeyframeFeatureCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr targetKeyframeFeatureCloud(new pcl::PointCloud<PointType>());
    pcl::io::loadPCDFile(ScansDirectory + std::to_string(_curr_kf_idx) + "_feature.pcd", *cureKeyframeFeatureCloud);
    pcl::io::loadPCDFile(ScansDirectory + std::to_string(_loop_kf_idx) + "_feature.pcd", *targetKeyframeFeatureCloud);


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

    Eigen::Affine3f transCur = pcl::getTransformation(0, 0, 0, 0, 0, _diff_yaw);

#if DEBUG_MODE_POSEGRAPH == 1
    sensor_msgs::PointCloud2 targetKeyframeCloudMsg2;
    pcl::toROSMsg(*targetKeyframeSurfCloud, targetKeyframeCloudMsg2);
    targetKeyframeCloudMsg2.header.frame_id = "/lidar_local";
    PubLoopTarget2.publish(targetKeyframeCloudMsg2);
#endif

    for (int i = 0; i < targetKeyframeCornerCloud->size(); ++i)
    {
        const auto &pointFrom = targetKeyframeCornerCloud->points[i];
        PointType tmp;
        tmp.x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z;
        tmp.y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z;
        tmp.z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z;
        tmp.intensity = pointFrom.intensity;
        tmp._PointXYZINormal::curvature = pointFrom._PointXYZINormal::curvature;
        tmp._PointXYZINormal::normal_y = pointFrom._PointXYZINormal::normal_y;
        tmp._PointXYZINormal::normal_z = pointFrom._PointXYZINormal::normal_z;
        targetKeyframeCornerCloud->points[i] = tmp;
    }

    for (int i = 0; i < targetKeyframeSurfCloud->size(); ++i)
    {
        const auto &pointFrom = targetKeyframeSurfCloud->points[i];
        PointType tmp;
        tmp.x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z;
        tmp.y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z;
        tmp.z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z;
        tmp.intensity = pointFrom.intensity;
        tmp._PointXYZINormal::curvature = pointFrom._PointXYZINormal::curvature;
        tmp._PointXYZINormal::normal_y = pointFrom._PointXYZINormal::normal_y;
        tmp._PointXYZINormal::normal_z = pointFrom._PointXYZINormal::normal_z;
        targetKeyframeSurfCloud->points[i] = tmp;
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
    for (size_t opti_counter = 0; opti_counter < 1; ++opti_counter)
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
        cout<<"cost : "<<final_cost<<endl;
    }

    if (final_cost < 0.02)
    {
        Eigen::Matrix3d rot = q_last_curr.toRotationMatrix();

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

        // Get pose transformation
        double roll, pitch, yaw;
        tf::Matrix3x3(tf::Quaternion(q_last_curr.x(), q_last_curr.y(), q_last_curr.z(), q_last_curr.w())).getRPY(roll, pitch, yaw);
        yaw += double(_diff_yaw);
        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(t_last_curr(0), t_last_curr(1), t_last_curr(2)));
        gtsam::Pose3 poseTo = Pose3(Rot3::RzRyRx(0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0));

        return poseFrom.between(poseTo);
    }
    else
    {
        return std::nullopt;
    }
}

std::optional<gtsam::Pose3> doICPVirtualRelative( int _loop_kf_idx, int _curr_kf_idx, float _diff_yaw )
{
    // parse pointclouds
    pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr targetKeyframeCloud(new pcl::PointCloud<PointType>());
    pcl::io::loadPCDFile(ScansDirectory + std::to_string(_curr_kf_idx) + "_full.pcd", *cureKeyframeCloud);
    pcl::io::loadPCDFile(ScansDirectory + std::to_string(_loop_kf_idx) + "_full.pcd", *targetKeyframeCloud);

    Eigen::Affine3f transCur = pcl::getTransformation(0, 0, 0, 0, 0, _diff_yaw);

    for (int i = 0; i < int(targetKeyframeCloud->size()); ++i)
    {
        const auto &pointFrom = targetKeyframeCloud->points[i];
        PointType tmp;
        tmp.x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
        tmp.y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
        tmp.z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
        tmp.intensity = pointFrom.intensity;
        targetKeyframeCloud->points[i] = tmp;
    }
    // ICP Settings
    pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(150); // giseop , use a value can cover 2*historyKeyframeSearchNum range in meter
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);

    // Align pointclouds
    icp.setInputSource(cureKeyframeCloud);
    icp.setInputTarget(targetKeyframeCloud);
    pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
    icp.align(*unused_result);

    float loopFitnessScoreThreshold = 0.3; // user parameter but fixed low value is safe.
    if (icp.hasConverged() == false || icp.getFitnessScore() > loopFitnessScoreThreshold)
    {
        std::cout << "[SC loop] ICP fitness test failed (" << icp.getFitnessScore() << " > " << loopFitnessScoreThreshold << "). Reject this SC loop." << std::endl;
        return std::nullopt;
    }
    else
    {
        std::cout << "[SC loop] ICP fitness test passed (" << icp.getFitnessScore() << " < " << loopFitnessScoreThreshold << "). Add this SC loop." << std::endl;
        Eigen::Matrix4f tf = icp.getFinalTransformation();
        pcl::transformPointCloud (*cureKeyframeCloud, *cureKeyframeCloud, tf);
    }

    // Get pose transformation
    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionLidarFrame;
    correctionLidarFrame = icp.getFinalTransformation();

    pcl::getTranslationAndEulerAngles (correctionLidarFrame, x, y, z, roll, pitch, yaw);
    gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
    gtsam::Pose3 poseTo = Pose3(Rot3::RzRyRx(0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0));

    return poseFrom.between(poseTo);
} // doICPVirtualRelative

void FrameHandler(const pd_loam::frameConstPtr &_frame)
{
    mBuf.lock();
    frameBuf.push(_frame);
    mBuf.unlock();
} // FrameHandler

void gpsHandler(const sensor_msgs::NavSatFixConstPtr &gps)
{
    Eigen::Vector3d llh;
    llh(0) = gps->latitude;
    llh(1) = gps->longitude;
    llh(2) = gps->altitude;
    Eigen::Vector3d xyz = llh2xyz(llh);
    if (gpsInitialized == false)
    {
        OriginXYZ = xyz;
        gpsInitialized = true;
    }
    gpsENU = xyz2enu(xyz,OriginXYZ);
    Eigen::Vector3d gpsCov;
    gpsCov(0) = gps->position_covariance[0];
    gpsCov(1) = gps->position_covariance[4];
    gpsCov(2) = gps->position_covariance[8];
    std::pair<Eigen::Vector3d, Eigen::Vector3d> gps_pair;
    gps_pair.first = gpsENU;
    gps_pair.second = gpsCov;
    gpsBuf.push(gps_pair);

    geometry_msgs::PoseStamped gpsPose;
    gpsPose.header.stamp = gps->header.stamp;
    gpsPose.header.frame_id = "/body";
    gpsPose.pose.position.x = gpsENU(0);
    gpsPose.pose.position.y = gpsENU(1);
    gpsPose.pose.position.z = gpsENU(2);

    gps_path.header.frame_id = "/body";
    gps_path.header.stamp = gps->header.stamp;
    gps_path.poses.push_back(gpsPose);
    PubPathGPS.publish(gps_path);
}

void imuOdomHandler(const nav_msgs::OdometryConstPtr &imuOdom)
{
    mBuf.lock();
    imuOdomBuf.push(imuOdom);
    mBuf.unlock();
}

void lidarOdomHandler(const nav_msgs::OdometryConstPtr &lidarOdom)
{
    mBuf.lock();
    lidarOdomBuf.push(lidarOdom);
    mBuf.unlock();
}

void gtPathHandler(const nav_msgs::PathConstPtr &gtPath)
{
    mBuf.lock();
    gt_path = *gtPath;
    mBuf.unlock();
}

void writeEdge(const std::pair<int, int> _node_idx_pair, const gtsam::Pose3& _relPose, std::vector<std::string>& edges_str)
{
    gtsam::Point3 t = _relPose.translation();
    gtsam::Rot3 R = _relPose.rotation();

    std::string curEdgeInfo {
        "EDGE_SE3:QUAT " + std::to_string(_node_idx_pair.first) + " " + std::to_string(_node_idx_pair.second) + " "
        + std::to_string(t.x()) + " " + std::to_string(t.y()) + " " + std::to_string(t.z())  + " "
        + std::to_string(R.toQuaternion().x()) + " " + std::to_string(R.toQuaternion().y()) + " "
        + std::to_string(R.toQuaternion().z()) + " " + std::to_string(R.toQuaternion().w()) };

    edges_str.emplace_back(curEdgeInfo);
}

void process_pg()
{
    while(1)
    {
        while (!frameBuf.empty())
        {
            mBuf.lock();

            if (frameBuf.empty())
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

            int frameIdx = frameBuf.front()->frame_idx;
            frameBuf.pop();
            Pose6D imuIncrement;
            if (!imuOdomBuf.empty())
            {
                imuIncrement = getOdom(*imuOdomBuf.front());
                imuOdomBuf.pop();
            }
            else
            {
                imuIncrement.x = 0; imuIncrement.y = 0; imuIncrement.z = 0;
                imuIncrement.roll = 0;  imuIncrement.pitch = 0; imuIncrement.yaw = 0;
            }

            std::pair<Eigen::Vector3d, Eigen::Vector3d> gpsmeas;
            if (!gpsBuf.empty())
            {
                gpsmeas = gpsBuf.front();
                gpsBuf.pop();
            }

            pcl::PointCloud<PointType>::Ptr thisFrameDS(new pcl::PointCloud<PointType>());
            downSizeFilterScancontext.setInputCloud(laserCloudFullRes);
            downSizeFilterScancontext.filter(*thisFrameDS);

            odom_pose_prev = odom_pose_curr;
            odom_pose_curr = pose_curr;
            Pose6D dtf = diffTransformation(odom_pose_prev, odom_pose_curr); // dtf means delta_transform

            double delta_translation = sqrt(dtf.x*dtf.x + dtf.y*dtf.y + dtf.z*dtf.z); // note: absolute value.
            translationAccumulated += delta_translation;
            rotaionAccumulated += (dtf.roll + dtf.pitch + dtf.yaw); // sum just naive approach.

            incrementAccumulated.x += imuIncrement.x;   incrementAccumulated.y += imuIncrement.y;   incrementAccumulated.z += imuIncrement.z;
            incrementAccumulated.roll += imuIncrement.roll; incrementAccumulated.pitch += imuIncrement.pitch;   incrementAccumulated.yaw += imuIncrement.yaw;
            if( translationAccumulated > keyframeMeterGap || rotaionAccumulated > keyframeRadGap )
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
            }
            else
            {
                isNowKeyFrame = false;
                //printf("pose graph not key frame time %f ms ++++++++++\n", t_pg.toc());
            }
            mBuf.unlock();

            if( ! isNowKeyFrame )   continue;

            mKF.lock();

            pcl::PointCloud<PointType>::Ptr laserCloudFeature (new pcl::PointCloud<PointType>());
            *laserCloudFeature += *laserCloudCorner;
            *laserCloudFeature += *laserCloudSurf;

            pcl::io::savePCDFileBinary(ScansDirectory + std::to_string(KeyFrameIdx) + "_full.pcd", *thisFrameDS);
            pcl::io::savePCDFileBinary(ScansDirectory + std::to_string(KeyFrameIdx) + "_feature.pcd", *laserCloudFeature);
            keyframePoses.push_back(pose_curr);
            keyframePosesUpdated.push_back(pose_curr);
            keyFrameIncrements.push_back(incrementAccumulated);
            keyframeTimes.push_back(timeLaserOdometry);

            incrementAccumulated.x = 0; incrementAccumulated.y = 0; incrementAccumulated.z = 0;
            incrementAccumulated.roll = 0;  incrementAccumulated.pitch = 0; incrementAccumulated.yaw = 0;

            scManager.makeAndSaveScancontextAndKeys(*thisFrameDS);

            mKF.unlock();

            const int prev_node_idx = keyframePoses.size() - 2;
            const int curr_node_idx = keyframePoses.size() - 1; // becuase cpp starts with 0 (actually this index could be any number, but for simple implementation, we follow sequential indexing)
            if( !gtSAMgraphMade /* prior node */)
            {
                const int init_node_idx = 0;
                gtsam::Pose3 poseOrigin = Pose6DtoGTSAMPose3(keyframePoses.at(init_node_idx));

                mPG.lock();                
                // prior factor
                gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(init_node_idx, poseOrigin, priorNoise));
                initialEstimate.insert(init_node_idx, poseOrigin);
                mPG.unlock();

                gtSAMgraphMade = true;
            }
            else /* consecutive node (and odom factor) after the prior added */
            {
                gtsam::Pose3 poseFrom = Pose6DtoGTSAMPose3(keyframePoses.at(prev_node_idx));
                gtsam::Pose3 poseTo = Pose6DtoGTSAMPose3(keyframePoses.at(curr_node_idx));

                mPG.lock();
                // odom factor
                gtsam::Pose3 relPose = poseFrom.between(poseTo);
                gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(prev_node_idx, curr_node_idx, relPose, odomNoise));

                if (keyFrameIncrements.at(curr_node_idx).x != 0 && keyFrameIncrements.at(curr_node_idx).y != 0 && keyFrameIncrements.at(curr_node_idx).z != 0)
                {
                    gtsam::Pose3 poseInc = Pose6DtoGTSAMPose3(keyFrameIncrements.at(curr_node_idx));
                    gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(prev_node_idx, curr_node_idx, poseInc, odomImuNoise));
                }
//                if (gpsInitialized == true && gpsmeas.second(0) < 2 && gpsmeas.second(1) < 2)
//                {
//                    std::cout<<"gps e : "<<gpsENU(0)<<", gps n :"<<gpsENU(1)<<std::endl;
//                    gpsmeas.second(0) = std::fmax(gpsmeas.second(0), 1);
//                    gpsmeas.second(1) = std::fmax(gpsmeas.second(1), 1);
//                    gpsmeas.second(2) = std::fmax(gpsmeas.second(2), 1);
//                    gpsNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(3) << pow(gpsmeas.second(0),2), pow(gpsmeas.second(1),2), pow(/*gpsmeas.second(2)*/100000,2)).finished()); // e,n,u
//                    gtsam::GPSFactor gps_factor(curr_node_idx, gtsam::Point3(gpsmeas.first(0), gpsmeas.first(1), poseTo.z()), gpsNoise);
//                    gtSAMgraph.add(gps_factor);
//                }
                initialEstimate.insert(curr_node_idx, poseTo);
                writeEdge({prev_node_idx, curr_node_idx}, relPose, edges_str);
                mPG.unlock();
            }
            KeyFrameIdx++;
            //printf("pose graph key frame time %f ms ++++++++++\n", t_pg.toc());
        }   //while (!frameBuf.empty())

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }   //while(1)
}   //process_pg()


void performSCLoopClosure(void)
{
    if( int(keyframePoses.size()) < scManager.NUM_EXCLUDE_RECENT || isNowKeyFrame == false) // do not try too early
        return;

    auto detectResult = scManager.detectLoopClosureID(); // first: nn index, second: yaw diff
    int SCclosestHistoryFrameID = detectResult.first;
    if( SCclosestHistoryFrameID != -1 )
    {
        if (detectResult.second > M_PI) detectResult.second -= 2*M_PI;

        const int prev_node_idx = SCclosestHistoryFrameID;
        const int curr_node_idx = keyframePoses.size() - 1;
        const float diff_yaw = detectResult.second;

        if (prev_node_idx != Prev_prev_node_idx && curr_node_idx != Prev_curr_node_idx)
        {
            //cout << "Loop detected! - between " << prev_node_idx << " and " << curr_node_idx << "" << endl;

            //cout<<"loop idx : "<<detectResult.first<<", diff yaw : "<<rad2deg(detectResult.second)<<endl;
            mBuf.lock();
            scLoopBuf.push(std::tuple<int, int, float>(prev_node_idx, curr_node_idx, diff_yaw));

            // addding actual 6D constraints in the other thread, PD-LOAM calculation.
            Prev_prev_node_idx = prev_node_idx;
            Prev_curr_node_idx = curr_node_idx;
            mBuf.unlock();
        }
    }
} // performSCLoopClosure

void process_lcd(void)
{
    float loopClosureFrequency = 5;
    ros::Rate rate(loopClosureFrequency);
    while (ros::ok())
    {
        rate.sleep();
        performSCLoopClosure();
    }
} // process_lcd

void process_loam(void)
{
    while(1)
    {
        while ( !scLoopBuf.empty() )
        {

            mloop.lock();
            std::tuple<int, int, float> loop_idx_pair = scLoopBuf.front();
            scLoopBuf.pop();

            const int prev_node_idx = get<0>(loop_idx_pair);
            const int curr_node_idx = get<1>(loop_idx_pair);
            const float diff_yaw = get<2>(loop_idx_pair);

//            TicToc t_icp;
//            auto relative_pose_optional = doICPVirtualRelative(prev_node_idx, curr_node_idx, diff_yaw);
//            printf("icp matching time %f ms ++++++++++\n", t_icp.toc());

            TicToc t_loam;
            auto relative_pose_optional = doLOAMVirtualRelative(prev_node_idx, curr_node_idx, diff_yaw);
            //printf("loam matching time %f ms ++++++++++\n", t_loam.toc());
            if(relative_pose_optional)
            {
                gtsam::Pose3 relative_pose = relative_pose_optional.value();
                gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(prev_node_idx, curr_node_idx, relative_pose, robustLoopNoise));
                writeEdge({prev_node_idx, curr_node_idx}, relative_pose, edges_str);
                loopLine.header.stamp = ros::Time().fromSec(timeLaserOdometry);
                geometry_msgs::Point p;
                p.x = keyframePoses[prev_node_idx].x;    p.y = keyframePoses[prev_node_idx].y;    p.z = keyframePoses[prev_node_idx].z;
                loopLine.points.push_back(p);
                p.x = keyframePoses[curr_node_idx].x;    p.y = keyframePoses[curr_node_idx].y;    p.z = keyframePoses[curr_node_idx].z;
                loopLine.points.push_back(p);
                PubLoopLineMarker.publish(loopLine);
                isLoopClosed = true;
            }
            mloop.unlock();
        }

        // wait (must required for running the while loop)
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
} // process_loam

void process_isam(void)
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
            pubPath();
            mPG.unlock();
            printf("pose graph optimization time %f ms ++++++++++\n", t_opt.toc());
        }
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "laserPG");
    ros::NodeHandle nh;

    // save directories
    nh.param<std::string>("save_directory", save_directory, "/"); // pose assignment every k m move
    nh.param<bool>("get_GT",getGT,false);

    ScansDirectory = save_directory + "Scans/";
    auto unused = system((std::string("exec rm -r ") + ScansDirectory).c_str());
    unused = system((std::string("mkdir -p ") + ScansDirectory).c_str());

    nh.param<double>("keyframe_meter_gap", keyframeMeterGap, 2.0); // pose assignment every k m move
    nh.param<double>("keyframe_deg_gap", keyframeDegGap, 10.0); // pose assignment every k deg rot
    keyframeRadGap = deg2rad(keyframeDegGap);

    nh.param<double>("sc_dist_thres", scDistThres, 0.2);
    nh.param<double>("sc_max_radius", scMaximumRadius, 80.0); // 80 is recommended for outdoor, and lower (ex, 20, 40) values are recommended for indoor
    nh.param<double>("sc_resolution", scResolution, 0.4);

    downSizeFilterScancontext.setLeafSize(scResolution, scResolution, scResolution);
    downSizeFilterICP.setLeafSize(scResolution, scResolution, scResolution);
    downSizeFilterMapPGO.setLeafSize(scResolution, scResolution, scResolution);

    scManager.setSCdistThres(scDistThres);
    scManager.setMaximumRadius(scMaximumRadius);

    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    isam = new ISAM2(parameters);
    initNoises();

    KF_Marker.type = visualization_msgs::Marker::SPHERE;
    KF_Marker.header.frame_id = "/body";
    KF_Marker.color.r = 1.0;
    KF_Marker.color.a = 1.0;
    KF_Marker.scale.x = 0.2;
    KF_Marker.scale.y = 0.2;
    KF_Marker.scale.z = 0.2;

    loopLine.type = visualization_msgs::Marker::LINE_LIST;
    loopLine.action = visualization_msgs::Marker::ADD;
    loopLine.color.g = 1.0; loopLine.color.a = 1.0;
    loopLine.scale.x = 0.05;
    loopLine.header.frame_id = "/body";

    nh.param<double>("matching_threshold",matchingThres, 4);

    ros::Subscriber subGPS = nh.subscribe<sensor_msgs::NavSatFix>("/ublox_gps/fix", 100, gpsHandler);

    ros::Subscriber subFrame = nh.subscribe<pd_loam::frame>("/mapping/frame", 100, FrameHandler);

    ros::Subscriber subImuOdom = nh.subscribe<nav_msgs::Odometry>("/imuPreintegration/imu_increment", 100, imuOdomHandler);

    ros::Subscriber subLidarOdom = nh.subscribe<nav_msgs::Odometry>("/mapping/aft_mapped_to_init", 100, lidarOdomHandler);

    ros::Subscriber subgtPath = nh.subscribe<nav_msgs::Path>("/path_gt", 100, gtPathHandler);

    PubKeyFrameMarker = nh.advertise<visualization_msgs::MarkerArray>("/posegraph/KF",100);

    PubLoopLineMarker = nh.advertise<visualization_msgs::Marker>("/posegraph/loopLine", 100);

    PubOdomAftPGO = nh.advertise<nav_msgs::Odometry>("/posegraph/OdomAftPG", 100);

    PubPathAftPGO = nh.advertise<nav_msgs::Path>("/posegraph/aft_PG_path", 100);
    PubPathPose = nh.advertise<nav_msgs::Path>("/posegraph/bf_PG_path",100);
    PubPathGT = nh.advertise<nav_msgs::Path>("/posegraph/GT_path",100);
    PubPathGPS = nh.advertise<nav_msgs::Path>("/gps_path", 100);
#if DEBUG_MODE_POSEGRAPH == 1
    PubLoopCurrent = nh.advertise<sensor_msgs::PointCloud2>("/loop_current", 100);
    PubLoopTarget = nh.advertise<sensor_msgs::PointCloud2>("/loop_target", 100);
    PubLoopTarget2 = nh.advertise<sensor_msgs::PointCloud2>("/loop_target_no_yaw", 100);
    PubLoopLOAM = nh.advertise<sensor_msgs::PointCloud2>("/loop_loam", 100);
#endif

    std::thread posegraph {process_pg}; // pose graph construction
    std::thread lc_detection {process_lcd}; // loop closure detection
    std::thread edge_calculation {process_loam};    //PD-LOAM based edge measurement calculation
    std::thread graph_optimization {process_isam};  //gtsam based pose graph optimization

    ros::spin();

    return 0;
}
