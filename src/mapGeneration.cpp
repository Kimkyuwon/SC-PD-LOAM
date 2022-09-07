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
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/kdtree.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Path.h>
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

using namespace std;
pcl::PointCloud<PointType>::Ptr laserCloudCorner(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurf(new pcl::PointCloud<PointType>());
std::vector<Eigen::Matrix3d> cornerCov;
std::vector<Eigen::Matrix3d> surfCov;
pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerMap(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfMap(new pcl::KdTreeFLANN<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudCornerPDMap(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfPDMap(new pcl::PointCloud<PointType>());
std::vector<Eigen::Matrix3d> cornerMapCov;
std::vector<Eigen::Matrix3d> surfMapCov;

double matchingThres;
string save_directory, Path_directory, ScansDirectory;
nav_msgs::Path PG_path;
std::mutex mBuf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

void getCornerProbabilityDistributions(pcl::PointCloud<PointType>::Ptr cornerTemp)
{
    int cnt = 0;
    pcl::PointCloud<PointType> tempPC;
    for (size_t k = 0; k < cornerTemp->points.size(); k++)
    {
        if (cornerTemp->points[k]._PointXYZINormal::normal_y == cnt)
        {
            tempPC.push_back(cornerTemp->points[k]);
            if (k == cornerTemp->points.size()-1)
            {
                Eigen::Vector3d mean = getMean(tempPC);
                Eigen::Matrix3d cov = getCovariance(tempPC);

                PointType tempPoint;
                tempPoint.x = mean(0);
                tempPoint.y = mean(1);
                tempPoint.z = mean(2);
                laserCloudCorner->push_back(tempPoint);
                cornerCov.push_back(cov);

                tempPC.clear();
                cnt++;
            }
        }
        else
        {
            Eigen::Vector3d mean = getMean(tempPC);
            Eigen::Matrix3d cov = getCovariance(tempPC);

            PointType tempPoint;
            tempPoint.x = mean(0);
            tempPoint.y = mean(1);
            tempPoint.z = mean(2);
            laserCloudCorner->push_back(tempPoint);
            cornerCov.push_back(cov);

            tempPC.clear();
            tempPC.push_back(cornerTemp->points[k]);
            cnt++;
        }
    }
}

void getSurfProbabilityDistributions(pcl::PointCloud<PointType>::Ptr surfTemp)
{
    int cnt = 0;
    pcl::PointCloud<PointType> tempPC;
    for (size_t k = 0; k < surfTemp->points.size(); k++)
    {
        if (surfTemp->points[k]._PointXYZINormal::normal_y == cnt)
        {
            tempPC.push_back(surfTemp->points[k]);

            if (k == surfTemp->points.size()-1)
            {
                Eigen::Vector3d mean = getMean(tempPC);
                Eigen::Matrix3d cov = getCovariance(tempPC);

                PointType tempPoint;
                tempPoint.x = mean(0);
                tempPoint.y = mean(1);
                tempPoint.z = mean(2);
                laserCloudSurf->push_back(tempPoint);
                surfCov.push_back(cov);

                tempPC.clear();
                cnt++;
            }
        }
        else
        {
            Eigen::Vector3d mean = getMean(tempPC);
            Eigen::Matrix3d cov = getCovariance(tempPC);

            PointType tempPoint;
            tempPoint.x = mean(0);
            tempPoint.y = mean(1);
            tempPoint.z = mean(2);

            laserCloudSurf->push_back(tempPoint);
            surfCov.push_back(cov);
            tempPC.clear();
            tempPC.push_back(surfTemp->points[k]);
            cnt++;
        }
    }
}

void updateMap()
{
    std::vector<int> MapSearchInd;
    std::vector<float> MapSearchSqDis;

    kdtreeCornerMap->setInputCloud(laserCloudCornerPDMap);
    for(size_t c = 0; c < laserCloudCorner->points.size(); c++)
    {
         kdtreeCornerMap->nearestKSearch(laserCloudCorner->points[c], 1, MapSearchInd, MapSearchSqDis);
         Eigen::Vector3d diff_mean;
         diff_mean(0) = laserCloudCorner->points[c].x - laserCloudCornerPDMap->points[MapSearchInd[0]].x;
         diff_mean(1) = laserCloudCorner->points[c].y - laserCloudCornerPDMap->points[MapSearchInd[0]].y;
         diff_mean(2) = laserCloudCorner->points[c].z - laserCloudCornerPDMap->points[MapSearchInd[0]].z;
         Eigen::Matrix3d mapCov = cornerMapCov[MapSearchInd[0]];
         double maha_dist = sqrt(diff_mean.transpose() * mapCov.inverse() * diff_mean);
         if (MapSearchSqDis[0] > matchingThres && maha_dist > 8)
         {
            laserCloudCornerPDMap->points.push_back(laserCloudCorner->points[c]);
            cornerMapCov.push_back(cornerCov[c]);
        }
    }

    kdtreeSurfMap->setInputCloud(laserCloudSurfPDMap);
    for(size_t c = 0; c < laserCloudSurf->points.size(); c++)
    {
        kdtreeSurfMap->nearestKSearch(laserCloudSurf->points[c], 1, MapSearchInd, MapSearchSqDis);
        Eigen::Vector3d diff_mean;
        diff_mean(0) = laserCloudSurf->points[c].x - laserCloudSurfPDMap->points[MapSearchInd[0]].x;
        diff_mean(1) = laserCloudSurf->points[c].y - laserCloudSurfPDMap->points[MapSearchInd[0]].y;
        diff_mean(2) = laserCloudSurf->points[c].z - laserCloudSurfPDMap->points[MapSearchInd[0]].z;
        Eigen::Matrix3d mapCov = surfMapCov[MapSearchInd[0]];
        double maha_dist = diff_mean.transpose() * mapCov.inverse() * diff_mean;
        if (MapSearchSqDis[0] > matchingThres && maha_dist > 8)
        {
            laserCloudSurfPDMap->points.push_back(laserCloudSurf->points[c]);
            surfMapCov.push_back(surfCov[c]);
        }
    }
}

void PoseGraphPathHandler(const nav_msgs::PathConstPtr &path)
{
    mBuf.lock();
    PG_path = *path;
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

        std::string PathInfo {to_string(tl(0)) + " " + to_string(tl(1)) + " " + to_string(tl(2)) + " "
            + to_string(q.x()) + " " + to_string(q.y()) + " " + to_string(q.z()) + " " + to_string(q.w())};
        stream << PathInfo << endl;
    }
    stream.close();
}

pcl::PointCloud<PointType> saveMap(nav_msgs::Path path)
{
    pcl::PointCloud<PointType>::Ptr map (new pcl::PointCloud<PointType> ());
    for (size_t i = 0; i < path.poses.size(); i++)
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
        Eigen::Matrix3d R = q.toRotationMatrix();
        Eigen::Matrix4d TF(Eigen::Matrix4d::Identity());
        TF.block(0,0,3,3) = R;
        TF(0,3) = tl(0);
        TF(1,3) = tl(1);
        TF(2,3) = tl(2);

        pcl::PointCloud<PointType> tempkeyframeLaserCloud;
        if(pcl::io::loadPCDFile(ScansDirectory + std::to_string(i) + "_full.pcd", tempkeyframeLaserCloud) == -1)
        {
            break;
        }

        pcl::PointCloud<PointType> keyframeLaserCloud;
        for (auto j = 0; j < tempkeyframeLaserCloud.size(); j++)
        {
            double r = sqrt(pow(tempkeyframeLaserCloud[j].x,2)+pow(tempkeyframeLaserCloud[j].y,2)+pow(tempkeyframeLaserCloud[j].z,2));
            if (r > 50) continue;
            keyframeLaserCloud.push_back(tempkeyframeLaserCloud[j]);
        }

        pcl::transformPointCloud(keyframeLaserCloud, keyframeLaserCloud, TF);
        *map += keyframeLaserCloud;
        downSizeFilterMap.setInputCloud(map);
        downSizeFilterMap.filter(*map);
    }
    return *map;
}

void savePDmap(nav_msgs::Path path)
{
    laserCloudCornerPDMap->clear();
    cornerMapCov.clear();
    laserCloudSurfPDMap->clear();
    surfMapCov.clear();
    for (size_t i = 0; i < path.poses.size(); i++)
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
        Eigen::Matrix3d R = q.toRotationMatrix();
        Eigen::Matrix4d TF(Eigen::Matrix4d::Identity());
        TF.block(0,0,3,3) = R;
        TF(0,3) = tl(0);
        TF(1,3) = tl(1);
        TF(2,3) = tl(2);

        laserCloudCorner->clear();
        cornerCov.clear();
        laserCloudSurf->clear();
        surfCov.clear();
        // parse pointclouds
        pcl::PointCloud<PointType> tempCloud;
        pcl::PointCloud<PointType>::Ptr FeatureCloud(new pcl::PointCloud<PointType>());
        pcl::io::loadPCDFile(ScansDirectory + std::to_string(i) + "_feature.pcd", tempCloud);
        for (auto j = 0; j < tempCloud.size(); j++)
        {
            double r = sqrt(pow(tempCloud[j].x,2)+pow(tempCloud[j].y,2)+pow(tempCloud[j].z,2));
            if (r > 50) continue;
            FeatureCloud->points.push_back(tempCloud[j]);
        }
        pcl::transformPointCloud(*FeatureCloud, *FeatureCloud, TF);

        pcl::PointCloud<PointType>::Ptr CornerCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr SurfCloud(new pcl::PointCloud<PointType>());

        for (int t = 0; t < FeatureCloud->points.size(); t++)
        {
            if (FeatureCloud->points[t]._PointXYZINormal::normal_z == 1)
            {
                CornerCloud->points.push_back(FeatureCloud->points[t]);
            }
            else if(FeatureCloud->points[t]._PointXYZINormal::normal_z == 2)
            {
                SurfCloud->points.push_back(FeatureCloud->points[t]);
            }
        }

        getCornerProbabilityDistributions(CornerCloud);

        getSurfProbabilityDistributions(SurfCloud);

        if (i == 0)
        {
            cout<<"surf size : "<<laserCloudSurf->points.size()<<endl;
            *laserCloudCornerPDMap = *laserCloudCorner;
            cornerMapCov = cornerCov;
            *laserCloudSurfPDMap = *laserCloudSurf;
            surfMapCov = surfCov;
            continue;
        }
        updateMap();
    }

    fstream stream = std::fstream(Path_directory + "pdMap.txt", std::fstream::out);
    for (auto k = 0; k < laserCloudCornerPDMap->points.size(); k++)
    {
        std::string PathInfo {to_string(1) + " " + to_string(laserCloudCornerPDMap->points[k].x) + " " + to_string(laserCloudCornerPDMap->points[k].y) + " " + to_string(laserCloudCornerPDMap->points[k].z) + " "
            + to_string(cornerMapCov[k](0,0)) + " " + to_string(cornerMapCov[k](0,1)) + " " + to_string(cornerMapCov[k](0,2)) + " "
            + to_string(cornerMapCov[k](1,1)) + " " + to_string(cornerMapCov[k](1,2)) + " " + to_string(cornerMapCov[k](2,2))};
        stream << PathInfo << endl;
    }
    for (auto k = 0; k < laserCloudSurfPDMap->points.size(); k++)
    {
        std::string PathInfo {to_string(2) + " " + to_string(laserCloudSurfPDMap->points[k].x) + " " + to_string(laserCloudSurfPDMap->points[k].y) + " " + to_string(laserCloudSurfPDMap->points[k].z) + " "
            + to_string(surfMapCov[k](0,0)) + " " + to_string(surfMapCov[k](0,1)) + " " + to_string(surfMapCov[k](0,2)) + " "
            + to_string(surfMapCov[k](1,1)) + " " + to_string(surfMapCov[k](1,2)) + " " + to_string(surfMapCov[k](2,2))};
        stream << PathInfo << endl;
    }
    stream.close();
    cout<<"map size : "<<laserCloudSurfPDMap->points.size()<<endl;
    pcl::io::savePCDFileBinary(Path_directory + "Surf_PD_map.pcd", *laserCloudSurfPDMap);
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

            savePath(PG_path, "stamped_pose_graph_estimate.txt");

            pcl::PointCloud<PointType> laserCloudPGMap = saveMap(PG_path);
            if(pcl::io::savePCDFileBinary(Path_directory + "PointCloud_AF_PG_Map.pcd", laserCloudPGMap) == -1)
            {
                ROS_ERROR("PGMap pcd file cannot save");
            }

            savePDmap(PG_path);

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
    ros::init(argc, argv, "mapGeneration");
    ros::NodeHandle nh;

    nh.param<double>("matching_threshold",matchingThres, 4);
    nh.param<std::string>("save_directory", save_directory, "/");

    Path_directory = save_directory + "trajectory/";
    auto unused = system((std::string("exec rm -r ") + Path_directory).c_str());
    unused = system((std::string("mkdir -p ") + Path_directory).c_str());
    ScansDirectory = save_directory + "Scans/";

    float mapRes = 0;
    nh.param<float>("mapping_resolution", mapRes, 0.5);
    downSizeFilterMap.setLeafSize(mapRes, mapRes, mapRes);

    ros::Subscriber subPoseGraphPath = nh.subscribe<nav_msgs::Path>("/posegraph/PGO_path", 100, PoseGraphPathHandler);

    std::thread saveFile{process_save};
    ros::spin();

    return 0;
}
