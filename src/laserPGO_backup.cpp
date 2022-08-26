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
#include <pcl/registration/ndt.h>
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

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include "common.h"
#include "tic_toc.h"
#include "lidarFactor.hpp"

#include "pd_loam/frame.h"
#include "pd_loam/gnss.h"

#define DEBUG_MODE_POSEGRAPH 1

std::string ScansDirectory, save_directory;

pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudCorner(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurf(new pcl::PointCloud<PointType>());

std::vector<Pose6D> keyframePoses;
std::vector<Pose6D> keyframePosesUpdated;
std::vector<double> keyframeTimes;
std::vector<Pose6D> keyFrameIncrements;

double keyframeMeterGap;
double keyframeDegGap, keyframeRadGap;
double translationAccumulated = 1000000.0; // large value means must add the first given frame.
double rotaionAccumulated = 1000000.0; // large value means must add the first given frame.

bool isNowKeyFrame = false;
bool poseInitialized = false;
bool isLoopClosed = false;
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

std::mutex mBuf, mKF, mloop;

visualization_msgs::Marker KF_Marker, loopLine;
visualization_msgs::MarkerArray KF_Markers, loopLines;

ros::Publisher PubKeyFrameMarker, PubLoopLineMarker;
ros::Publisher PubLoopCurrent, PubLoopTarget, PubLoopLOAM;

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


std::optional<Eigen::Matrix4d> doLOAMVirtualRelative( int _loop_kf_idx, int _curr_kf_idx, Eigen::Matrix4d _diff_TF )
{
    // parse pointclouds
    pcl::PointCloud<PointType>::Ptr cureKeyframeFeatureCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr targetKeyframeFeatureCloud(new pcl::PointCloud<PointType>());
    for (int i = -2; i <=2; i++)
    {
        if (_curr_kf_idx+i > 0 && _curr_kf_idx+i < keyframePoses.size())
        {
            pcl::PointCloud<PointType>::Ptr temp_cureKeyframeFeatureCloud(new pcl::PointCloud<PointType>());
            pcl::io::loadPCDFile(ScansDirectory + std::to_string(_curr_kf_idx+i) + "_full.pcd", *temp_cureKeyframeFeatureCloud);
            Pose6D temp_pose = keyframePoses[_curr_kf_idx+i];
            Eigen::Matrix4d curr_TF = get_TF_Matrix(keyframePoses.back());
            Eigen::Matrix4d temp_TF = get_TF_Matrix(temp_pose);
            Eigen::Matrix4d temp_diffTF = curr_TF.inverse()*temp_TF;
            pcl::transformPointCloud(*temp_cureKeyframeFeatureCloud,*temp_cureKeyframeFeatureCloud,temp_diffTF);
            *cureKeyframeFeatureCloud += *temp_cureKeyframeFeatureCloud;
        }
        if (_loop_kf_idx+i > 0 && _loop_kf_idx+i < keyframePoses.size())
        {
            pcl::PointCloud<PointType>::Ptr temp_targetKeyframeFeatureCloud(new pcl::PointCloud<PointType>());
            pcl::io::loadPCDFile(ScansDirectory + std::to_string(_loop_kf_idx+i) + "_full.pcd", *temp_targetKeyframeFeatureCloud);
            Pose6D temp_pose = keyframePoses[_loop_kf_idx+i];
            Eigen::Matrix4d curr_TF = get_TF_Matrix(keyframePoses[_loop_kf_idx]);
            Eigen::Matrix4d temp_TF = get_TF_Matrix(temp_pose);
            Eigen::Matrix4d temp_diffTF = curr_TF.inverse()*temp_TF;
            pcl::transformPointCloud(*temp_targetKeyframeFeatureCloud,*temp_targetKeyframeFeatureCloud,temp_diffTF);
            *targetKeyframeFeatureCloud += *temp_targetKeyframeFeatureCloud;
        }
    }
    //pcl::transformPointCloud(*targetKeyframeFeatureCloud, *targetKeyframeFeatureCloud, _diff_TF);

    pcl::VoxelGrid<PointType> voxel;
    voxel.setLeafSize(0.2,0.2,0.2);
    voxel.setInputCloud(cureKeyframeFeatureCloud);
    voxel.filter(*cureKeyframeFeatureCloud);
    voxel.setInputCloud(targetKeyframeFeatureCloud);
    voxel.filter(*targetKeyframeFeatureCloud);
#if DEBUG_MODE_POSEGRAPH == 1
    // loop verification
    sensor_msgs::PointCloud2 cureKeyframeCloudMsg;
    pcl::toROSMsg(*cureKeyframeFeatureCloud, cureKeyframeCloudMsg);
    cureKeyframeCloudMsg.header.frame_id = "/lidar_local";
    PubLoopCurrent.publish(cureKeyframeCloudMsg);

    sensor_msgs::PointCloud2 targetKeyframeCloudMsg;
    pcl::toROSMsg(*targetKeyframeFeatureCloud, targetKeyframeCloudMsg);
    targetKeyframeCloudMsg.header.frame_id = "/lidar_local";
    PubLoopTarget.publish(targetKeyframeCloudMsg);
#endif
    pclomp::NormalDistributionsTransform<PointType, PointType> ndt;
    ndt.setResolution(2.0);
    ndt.setMaximumIterations(5);
    ndt.setStepSize(0.1);
    ndt.setTransformationEpsilon(0.01);
    ndt.setNumThreads(4);
    ndt.setNeighborhoodSearchMethod(pclomp::DIRECT7);
    ndt.setInputSource(cureKeyframeFeatureCloud);
    ndt.setInputTarget(targetKeyframeFeatureCloud);
    pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
    ndt.align(*unused_result, _diff_TF.cast<float>());

    Eigen::Matrix4f result_TF = ndt.getFinalTransformation();
    if (ndt.hasConverged() == true /*< 0.02*/)
    {
        pcl::transformPointCloud(*cureKeyframeFeatureCloud, *cureKeyframeFeatureCloud, result_TF);

#if DEBUG_MODE_POSEGRAPH == 1
        sensor_msgs::PointCloud2 LOAMCloudMsg;
        pcl::toROSMsg(*cureKeyframeFeatureCloud, LOAMCloudMsg);
        LOAMCloudMsg.header.frame_id = "/lidar_local";
        PubLoopLOAM.publish(LOAMCloudMsg);
#endif

        // Get pose transformation
        Eigen::Matrix4d FromTo_TF(Eigen::Matrix4d::Identity());
        FromTo_TF = result_TF.cast<double>();

        return FromTo_TF;
    }
    else
    {
        return std::nullopt;
    }
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

            pcl::io::savePCDFileBinary(ScansDirectory + std::to_string(KeyFrameIdx) + "_full.pcd", *laserCloudFullRes);
            pcl::io::savePCDFileBinary(ScansDirectory + std::to_string(KeyFrameIdx) + "_feature.pcd", *laserCloudFeature);
            keyframePoses.push_back(pose_curr);
            keyframePosesUpdated.push_back(pose_curr);
            keyFrameIncrements.push_back(incrementAccumulated);
            keyframeTimes.push_back(timeLaserOdometry);

            incrementAccumulated.x = 0; incrementAccumulated.y = 0; incrementAccumulated.z = 0;
            incrementAccumulated.roll = 0;  incrementAccumulated.pitch = 0; incrementAccumulated.yaw = 0;
            KeyFrameIdx++;

            Pose6D To_pose = keyframePoses.back();
            Eigen::Matrix4d to_TF = get_TF_Matrix(To_pose);

            std::vector<float> LOOP_DIST;
            std::vector<int> LOOP_IDX;
            std::vector<Eigen::Matrix4d> FROM_TFs;
            if (keyframePoses.size() < PassIdx)
            {
                mKF.unlock();
                continue;
            }

            for (size_t i = 0; i < keyframePoses.size() - PassIdx; i++)
            {
                Pose6D From_pose = keyframePoses[i];
                Eigen::Matrix4d from_TF = get_TF_Matrix(From_pose);

                float loop_dist = sqrt(pow((to_TF(0,3)-from_TF(0,3)),2)+pow((to_TF(1,3)-from_TF(1,3)),2)+pow((to_TF(2,3)-from_TF(2,3)),2));

                if (loop_dist > 10)  continue;
                LOOP_DIST.push_back(loop_dist);
                LOOP_IDX.push_back(i);
                FROM_TFs.push_back(from_TF);
            }
            if (LOOP_DIST.size() > 0)
            {
                size_t min_idx = 0;
                float min_loop_dist = LOOP_DIST.front();
                for (size_t k = 1; k < LOOP_DIST.size(); k++)
                {
                    Eigen::Matrix4d temp_TF = FROM_TFs[k].inverse() * to_TF;
                    if (min_loop_dist > LOOP_DIST[k] && temp_TF(0,3) > 8)
                    {
                        min_loop_dist = LOOP_DIST[k];
                        min_idx = k;
                    }
                }
                ROS_INFO("loop detected!!");
                loopLine.header.stamp = ros::Time().fromSec(timeLaserOdometry);
                geometry_msgs::Point p;
                p.x = FROM_TFs[min_idx](0,3);    p.y = FROM_TFs[min_idx](1,3);    p.z = FROM_TFs[min_idx](2,3);
                loopLine.points.push_back(p);
                p.x = to_TF(0,3);    p.y = to_TF(1,3);    p.z = to_TF(2,3);
                loopLine.points.push_back(p);
                PubLoopLineMarker.publish(loopLine);

                Eigen::Matrix4d delta_TF = FROM_TFs[min_idx].inverse() * to_TF;

                mBuf.lock();
                LoopBuf.push(std::tuple<int, int, Eigen::Matrix4d>(LOOP_IDX[min_idx], keyframePoses.size()-1, delta_TF));
                mBuf.unlock();
            }
            mKF.unlock();
        }
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

void process_loam(void)
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
//                gtsam::Pose3 relative_pose = relative_pose_optional.value();
//                gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(prev_node_idx, curr_node_idx, relative_pose, robustLoopNoise));
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

    PubKeyFrameMarker = nh.advertise<visualization_msgs::MarkerArray>("/posegraph/KF",100);

    PubLoopLineMarker = nh.advertise<visualization_msgs::Marker>("/posegraph/loopLine", 100);
#if DEBUG_MODE_POSEGRAPH == 1
    PubLoopCurrent = nh.advertise<sensor_msgs::PointCloud2>("/loop_current", 100);
    PubLoopTarget = nh.advertise<sensor_msgs::PointCloud2>("/loop_target", 100);
    PubLoopLOAM = nh.advertise<sensor_msgs::PointCloud2>("/loop_loam", 100);
#endif

    std::thread posegraph {process_pg}; // pose graph construction
    std::thread edge_calculation {process_loam};    //PD-LOAM based edge measurement calculation
    ros::spin();

    return 0;
}
