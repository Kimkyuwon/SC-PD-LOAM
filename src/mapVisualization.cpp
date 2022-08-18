#include <math.h>
#include <vector>
#include <fstream>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/registration/transforms.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Path.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <string>

#include "common.h"
#include "tic_toc.h"

#include "pd_loam/frame.h"

using namespace std;
int mapVizRange = 0;
string save_directory, Path_directory, ScansDirectory;
string evaluation_type;
bool getGT = false;
nav_msgs::Path map_path, PG_path, GT_path;
//Mapping result
pcl::PointCloud<PointType>::Ptr laserCloudSubMap(new pcl::PointCloud<PointType>());
std::queue<pd_loam::frameConstPtr> frameBuf;
std::mutex mBuf;
std::mutex mFrame;
pcl::VoxelGrid<PointType> downSizeFilterMap;

ros::Publisher pubLaserCloudSurround, pubLaserCloudSubMap, pubLaserCloudMap, pubLaserCloudPGMap;
void FrameHandler(const pd_loam::frameConstPtr &_frame)
{
    mBuf.lock();
    frameBuf.push(_frame);
    ros::Time timeFrame = frameBuf.front()->header.stamp;
    int frameCount = frameBuf.front()->frame_idx;
    nav_msgs::Odometry framePose = frameBuf.front()->pose;

    Eigen::Quaterniond q_frame;
    q_frame.w() = framePose.pose.pose.orientation.w;
    q_frame.x() = framePose.pose.pose.orientation.x;
    q_frame.y() = framePose.pose.pose.orientation.y;
    q_frame.z() = framePose.pose.pose.orientation.z;
    Eigen::Matrix3d R_frame = q_frame.toRotationMatrix();
    Eigen::Matrix4d TF_frame(Eigen::Matrix4d::Identity());
    TF_frame.block(0,0,3,3) = R_frame;
    TF_frame(0,3) = framePose.pose.pose.position.x;
    TF_frame(1,3) = framePose.pose.pose.position.y;
    TF_frame(2,3) = framePose.pose.pose.position.z;


    if (frameCount % 2 == 0)
    {
        pcl::PointCloud<PointType>::Ptr laserCloudFullRes (new pcl::PointCloud<PointType>());
        pcl::fromROSMsg(frameBuf.front()->fullPC, *laserCloudFullRes);
        pcl::transformPointCloud(*laserCloudFullRes,*laserCloudFullRes,TF_frame);
        *laserCloudSubMap += *laserCloudFullRes;

        downSizeFilterMap.setInputCloud(laserCloudSubMap);
        downSizeFilterMap.filter(*laserCloudSubMap);

        pcl::ConditionAnd<PointType>::Ptr rangeCondition (new pcl::ConditionAnd<PointType>());
        pcl::PointCloud<PointType>::Ptr filtered_map_ptr (new pcl::PointCloud<PointType>());
        rangeCondition->addComparison (pcl::FieldComparison<PointType>::ConstPtr (new pcl::FieldComparison<PointType> ("x", pcl::ComparisonOps::GT, TF_frame(0,3) -mapVizRange)));
        rangeCondition->addComparison (pcl::FieldComparison<PointType>::ConstPtr (new pcl::FieldComparison<PointType> ("x", pcl::ComparisonOps::LT, TF_frame(0,3) +mapVizRange)));
        rangeCondition->addComparison (pcl::FieldComparison<PointType>::ConstPtr (new pcl::FieldComparison<PointType> ("y", pcl::ComparisonOps::GT, TF_frame(1,3) -mapVizRange)));
        rangeCondition->addComparison (pcl::FieldComparison<PointType>::ConstPtr (new pcl::FieldComparison<PointType> ("y", pcl::ComparisonOps::LT, TF_frame(1,3) +mapVizRange)));
        pcl::ConditionalRemoval<PointType> ConditionRemoval;
        ConditionRemoval.setCondition (rangeCondition);
        ConditionRemoval.setInputCloud (laserCloudSubMap);
        ConditionRemoval.filter (*laserCloudSubMap);

        sensor_msgs::PointCloud2 laserCloudMsg;
        pcl::toROSMsg(*laserCloudSubMap, laserCloudMsg);
        laserCloudMsg.header.stamp = timeFrame;
        laserCloudMsg.header.frame_id = "/body";
        pubLaserCloudSubMap.publish(laserCloudMsg);
    }
    if (frameCount % 1 == 0)
    {
        pcl::PointCloud<PointType>::Ptr laserCloudSurround(new pcl::PointCloud<PointType>());
        pcl::ConditionAnd<PointType>::Ptr rangeCondition (new pcl::ConditionAnd<PointType> ());
        pcl::PointCloud<PointType>::Ptr filtered_map_ptr (new pcl::PointCloud<PointType>());
        rangeCondition->addComparison (pcl::FieldComparison<PointType>::ConstPtr (new pcl::FieldComparison<PointType> ("x", pcl::ComparisonOps::GT, TF_frame(0,3) -50)));
        rangeCondition->addComparison (pcl::FieldComparison<PointType>::ConstPtr (new pcl::FieldComparison<PointType> ("x", pcl::ComparisonOps::LT, TF_frame(0,3) +50)));
        rangeCondition->addComparison (pcl::FieldComparison<PointType>::ConstPtr (new pcl::FieldComparison<PointType> ("y", pcl::ComparisonOps::GT, TF_frame(1,3) -50)));
        rangeCondition->addComparison (pcl::FieldComparison<PointType>::ConstPtr (new pcl::FieldComparison<PointType> ("y", pcl::ComparisonOps::LT, TF_frame(1,3) +50)));
        pcl::ConditionalRemoval<PointType> ConditionRemoval;
        ConditionRemoval.setCondition (rangeCondition);
        ConditionRemoval.setInputCloud (laserCloudSubMap);
        ConditionRemoval.filter (*laserCloudSurround);

        sensor_msgs::PointCloud2 laserCloudSurround3;
        pcl::toROSMsg(*laserCloudSurround, laserCloudSurround3);
        laserCloudSurround3.header.stamp = timeFrame;
        laserCloudSurround3.header.frame_id = "/body";
        pubLaserCloudSurround.publish(laserCloudSurround3);
    }
    frameBuf.pop();
    mBuf.unlock();
} // FrameHandler

void MapPathHandler(const nav_msgs::PathConstPtr &path)
{
    mBuf.lock();
    map_path = *path;
    mBuf.unlock();
}

void PoseGraphPathHandler(const nav_msgs::PathConstPtr &path)
{
    mBuf.lock();
    PG_path = *path;
    mBuf.unlock();
}

void gtPathHandler(const nav_msgs::PathConstPtr &path)
{
    mBuf.lock();
    GT_path = *path;
    mBuf.unlock();
}

void savePath(nav_msgs::Path path, string fileName)
{
    fstream stream = std::fstream(Path_directory + fileName, std::fstream::out);
    for (int i = 0; i < int(path.poses.size()); i++)
    {
        Eigen::Vector3d tl;
        Eigen::Quaterniond q;
        q.w() = path.poses[i].pose.orientation.w;
        q.x() = path.poses[i].pose.orientation.x;
        q.y() = path.poses[i].pose.orientation.y;
        q.z() = path.poses[i].pose.orientation.z;
        tl(0) = path.poses[i].pose.position.x;
        tl(1) = path.poses[i].pose.position.y;
        tl(2) = path.poses[i].pose.position.z;

        if (evaluation_type == "evo")
        {
            Eigen::Matrix3d R = q.toRotationMatrix();
            Eigen::Matrix4d TF(Eigen::Matrix4d::Identity());
            TF.block(0,0,3,3) = R;
            TF(0,3) = tl(0);
            TF(1,3) = tl(1);
            TF(2,3) = tl(2);
            std::string TFInfo {
                to_string(TF(0,0)) + " " + to_string(TF(0,1)) + " " + to_string(TF(0,2)) + " " + to_string(TF(0,3)) + " "
                + to_string(TF(1,0)) + " " + to_string(TF(1,1)) + " " + to_string(TF(1,2)) + " " + to_string(TF(1,3)) + " "
                + to_string(TF(2,0)) + " " + to_string(TF(2,1)) + " " + to_string(TF(2,2)) + " " + to_string(TF(2,3))};
            stream << TFInfo << endl;
        }
        else if (evaluation_type == "rpg")
        {
            std::string PathInfo {to_string(path.poses[i].header.stamp.toSec()) + " " +
                to_string(tl(0)) + " " + to_string(tl(1)) + " " + to_string(tl(2)) + " "
                + to_string(q.x()) + " " + to_string(q.y()) + " " + to_string(q.z()) + " " + to_string(q.w())};
            stream << PathInfo << endl;
        }
    }
    stream.close();
}

pcl::PointCloud<PointType> saveMap(nav_msgs::Path path)
{
    pcl::PointCloud<PointType>::Ptr map (new pcl::PointCloud<PointType> ());
    for (size_t i = 0; i < path.poses.size(); i++)
    {
        if (i % 10 != 0) continue;
        Eigen::Vector3d tl;
        Eigen::Quaterniond q;
        q.w() = path.poses[i].pose.orientation.w;
        q.x() = path.poses[i].pose.orientation.x;
        q.y() = path.poses[i].pose.orientation.y;
        q.z() = path.poses[i].pose.orientation.z;
        tl(0) = path.poses[i].pose.position.x;
        tl(1) = path.poses[i].pose.position.y;
        tl(2) = path.poses[i].pose.position.z;
        Eigen::Matrix3d R = q.toRotationMatrix();
        Eigen::Matrix4d TF(Eigen::Matrix4d::Identity());
        TF.block(0,0,3,3) = R;
        TF(0,3) = tl(0);
        TF(1,3) = tl(1);
        TF(2,3) = tl(2);

        pcl::PointCloud<PointType> keyframeLaserCloud;
        if(pcl::io::loadPCDFile(ScansDirectory + std::to_string(i) + "_full.pcd", keyframeLaserCloud) == -1)
        {
            break;
        }
        pcl::transformPointCloud(keyframeLaserCloud, keyframeLaserCloud, TF);
        *map += keyframeLaserCloud;
        downSizeFilterMap.setInputCloud(map);
        downSizeFilterMap.filter(*map);
    }
    return *map;
}

void process_save()
{
    while(1)
    {
        char c = getchar();
        if (c == 's')
        {
            ROS_INFO("Save trajectory and map pcd files.");
            mBuf.lock();
            savePath(map_path, "stamped_traj_estimate.txt");

            savePath(PG_path, "stamped_pose_graph_estimate.txt");

            if (getGT == true)
            {
                savePath(GT_path, "stamped_groundtruth.txt");
            }

            pcl::PointCloud<PointType> laserCloudMap = saveMap(map_path);
            if(pcl::io::savePCDFileBinary(Path_directory + "PointCloudMap.pcd", laserCloudMap) == -1)
            {
                ROS_ERROR("Map pcd file cannot save");
            }
            pcl::PointCloud<PointType> laserCloudPGMap = saveMap(PG_path);
            if(pcl::io::savePCDFileBinary(Path_directory + "PointCloud_AF_PG_Map.pcd", laserCloudPGMap) == -1)
            {
                ROS_ERROR("PGMap pcd file cannot save");
            }

            sensor_msgs::PointCloud2 laserCloudMapMsg;
            pcl::toROSMsg(laserCloudMap, laserCloudMapMsg);
            laserCloudMapMsg.header.stamp = map_path.poses.back().header.stamp;
            laserCloudMapMsg.header.frame_id = "/body";
            pubLaserCloudMap.publish(laserCloudMapMsg);

            sensor_msgs::PointCloud2 laserCloudPGMapMsg;
            pcl::toROSMsg(laserCloudPGMap, laserCloudPGMapMsg);
            laserCloudPGMapMsg.header.stamp = map_path.poses.back().header.stamp;
            laserCloudPGMapMsg.header.frame_id = "/body";
            pubLaserCloudPGMap.publish(laserCloudPGMapMsg);

            mBuf.unlock();
            ROS_INFO("Trajectory and map pcd files are saved.");
        }
        // wait (must required for running the while loop)
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "mapVisualization");
    ros::NodeHandle nh;

    nh.param<std::string>("save_directory", save_directory, "/");
    nh.param<std::string>("evaluation_type",evaluation_type,"rpg");
    nh.param<bool>("get_GT",getGT,false);

    Path_directory = save_directory + "trajectory/";
    auto unused = system((std::string("exec rm -r ") + Path_directory).c_str());
    unused = system((std::string("mkdir -p ") + Path_directory).c_str());
    ScansDirectory = save_directory + "Scans/";

    float mapRes = 0;
    nh.param<float>("mapping_resolution", mapRes, 0.5);
    nh.param<int>("map_viz_range",mapVizRange, 300);
    downSizeFilterMap.setLeafSize(mapRes, mapRes, mapRes);

    ros::Subscriber subFrame = nh.subscribe<pd_loam::frame>("/mapping/frame", 100, FrameHandler);

    ros::Subscriber subMapPath = nh.subscribe<nav_msgs::Path>("/posegraph/bf_PG_path", 100, MapPathHandler);

    ros::Subscriber subPoseGraphPath = nh.subscribe<nav_msgs::Path>("/posegraph/aft_PG_path", 100, PoseGraphPathHandler);

    ros::Subscriber subgtPath = nh.subscribe<nav_msgs::Path>("/posegraph/GT_path", 100, gtPathHandler);

    pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("/mapping/laser_cloud_surround", 100);

    pubLaserCloudSubMap = nh.advertise<sensor_msgs::PointCloud2>("/mapping/laser_cloud_submap", 100);

    pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/mapping/bf_PG_map", 100);

    pubLaserCloudPGMap = nh.advertise<sensor_msgs::PointCloud2>("/posegraph/aft_PG_map", 100);
    std::thread saveFile{process_save};
    ros::spin();

    return 0;
}
