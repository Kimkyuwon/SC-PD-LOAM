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

std::string map_directory;
std::ifstream readFile;

// every map points of probability distribution
pcl::PointCloud<PointType>::Ptr laserCloudCornerPDMap(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfPDMap(new pcl::PointCloud<PointType>());

std::vector<Eigen::Matrix3d> cornerMapCov;
std::vector<Eigen::Matrix3d> surfMapCov;

int marker_id = 1;
visualization_msgs::MarkerArray eig_marker;

ros::Publisher pubPointCloudMap, pubMapPath, pubEigenMarker;

void drawEigenVector(ros::Time time, int type, int marker_id, Eigen::Vector3d mean, Eigen::Matrix3d eigVec, Eigen::Vector3d eigen_value, visualization_msgs::MarkerArray &markerarr)
{
    for (int e = 0; e < 3; e++)
    {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "/body";
        marker.header.stamp = time;
        marker.type = visualization_msgs::Marker::ARROW;
        marker.action = visualization_msgs::Marker::ADD;
        marker.color.a = 1;
        marker.scale.x = 0.05;
        marker.scale.y = 0.05;
        marker.scale.z = 0.05;
        if (type == 1)      marker.color.r = 1;
        else if (type == 2)      marker.color.b = 1;
        //marker.lifetime = ros::Duration(0.2);
        geometry_msgs::Point p;
        p.x = mean(0);    p.y = mean(1);    p.z = mean(2);
        marker.points.push_back(p);
        Eigen::Vector3d ev = eigVec.col(e);
        p.x = mean(0)+sqrt(eigen_value(e))*ev(0);    p.y = mean(1)+sqrt(eigen_value(e))*ev(1);    p.z = mean(2)+sqrt(eigen_value(e))*ev(2);
        marker.points.push_back(p);
        marker.id = marker_id;
        markerarr.markers.push_back(marker);
        marker_id++;

        visualization_msgs::Marker marker2;
        marker2.header.frame_id = "/body";
        marker2.header.stamp = time;
        marker2.type = visualization_msgs::Marker::ARROW;
        marker2.action = visualization_msgs::Marker::ADD;
        marker2.color.a = 1;
        marker2.scale.x = 0.05;
        marker2.scale.y = 0.05;
        marker2.scale.z = 0.05;
        if (type == 1)      marker2.color.r = 1;
        else if (type == 2)      marker2.color.b = 1;
        //marker.lifetime = ros::Duration(0.2);
        p.x = mean(0);    p.y = mean(1);    p.z = mean(2);
        marker2.points.push_back(p);
        p.x = mean(0)-sqrt(eigen_value(e))*ev(0);    p.y = mean(1)-sqrt(eigen_value(e))*ev(1);    p.z = mean(2)-sqrt(eigen_value(e))*ev(2);
        marker2.points.push_back(p);
        marker2.id = marker_id;
        markerarr.markers.push_back(marker2);
        marker_id++;
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "mapViewer");
    ros::NodeHandle nh;

    nh.param<std::string>("map_directory", map_directory, "/");

    pcl::PointCloud<PointType>::Ptr PointCloudMap (new pcl::PointCloud<PointType>());
    pcl::io::loadPCDFile(map_directory + "PointCloud_AF_PG_Map.pcd", *PointCloudMap);

    std::string filePath = map_directory + "stamped_pose_graph_estimate.txt";
    readFile.open(filePath);
    std::string line;
    nav_msgs::Path MapPath;
    if(readFile.is_open())
    {
        while(std::getline(readFile, line))
        {
            std::stringstream sstream(line);
            std::string word;
            std::vector<float> MapData;
            while(getline(sstream, word, ' '))
            {
                MapData.push_back(std::stod(word));
            }
            geometry_msgs::PoseStamped posestamped;
            posestamped.header.stamp = ros::Time::now();
            posestamped.header.frame_id = "/body";
            posestamped.pose.position.x = MapData[0];
            posestamped.pose.position.y = MapData[1];
            posestamped.pose.position.z = MapData[2];
            posestamped.pose.orientation.w = MapData[6];
            posestamped.pose.orientation.x = MapData[3];
            posestamped.pose.orientation.y = MapData[4];
            posestamped.pose.orientation.z = MapData[5];

            MapPath.header.stamp = ros::Time::now();
            MapPath.header.frame_id = "/body";
            MapPath.poses.push_back(posestamped);
        }
        readFile.close();
    }
    else
    {
        std::cout << "Unable to open file";
        return 1;
    }

    std::string mapPath = map_directory + "pdMap.txt";
    readFile.open(mapPath);
    if(readFile.is_open())
    {
        while(std::getline(readFile, line))
        {
            std::stringstream sstream(line);
            std::string word;
            std::vector<double> MapData;
            while(getline(sstream, word, ' '))
            {
                MapData.push_back(std::stod(word));
            }
            int map_type = MapData[0];
            PointType p;
            p.x = MapData[1];
            p.y = MapData[2];
            p.z = MapData[3];
            Eigen::Matrix3d cov;
            cov(0,0) = MapData[4];  cov(0,1) = MapData[5];  cov(0,2) = MapData[6];
            cov(1,0) = cov(0,1);    cov(1,1) = MapData[7];  cov(1,2) = MapData[8];
            cov(2,0) = cov(0,2);    cov(2,1) = cov(1,2);  cov(2,2) = MapData[9];

            if (map_type == 1)
            {
                laserCloudCornerPDMap->points.push_back(p);
                cornerMapCov.push_back(cov);
            }
            else if (map_type == 2)
            {
                laserCloudSurfPDMap->points.push_back(p);
                surfMapCov.push_back(cov);
            }
        }
        readFile.close();
    }
    else
    {
        std::cout << "Unable to open file";
        return 1;
    }

    for (size_t c = 0; c < cornerMapCov.size(); c++)
    {
        if (c % 10 != 0)    continue;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cornerMapCov[c]);
        Eigen::Vector3d mean;
        mean(0) = laserCloudCornerPDMap->points[c].x;
        mean(1) = laserCloudCornerPDMap->points[c].y;
        mean(2) = laserCloudCornerPDMap->points[c].z;
        drawEigenVector(ros::Time::now(), 1, marker_id, mean, saes.eigenvectors(), saes.eigenvalues(), eig_marker);
        marker_id = eig_marker.markers.back().id+1;
    }

    for (size_t s = 0; s < surfMapCov.size(); s++)
    {
        if (s % 10 != 0)    continue;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(surfMapCov[s]);
        Eigen::Vector3d mean;
        mean(0) = laserCloudSurfPDMap->points[s].x;
        mean(1) = laserCloudSurfPDMap->points[s].y;
        mean(2) = laserCloudSurfPDMap->points[s].z;
        drawEigenVector(ros::Time::now(), 2, marker_id, mean, saes.eigenvectors(), saes.eigenvalues(), eig_marker);
        marker_id = eig_marker.markers.back().id+1;
    }

    float mapRes = 0;
    pcl::PointCloud<PointType>::Ptr FilteredMap (new pcl::PointCloud<PointType>());
    nh.param<float>("mapping_resolution", mapRes, 0.5);
    pcl::VoxelGrid<PointType> downSizeFilterMap;
    downSizeFilterMap.setLeafSize(mapRes, mapRes, mapRes);
    downSizeFilterMap.setInputCloud(PointCloudMap);
    downSizeFilterMap.filter(*FilteredMap);
    sensor_msgs::PointCloud2 pointMapMsg;
    pcl::toROSMsg(*FilteredMap, pointMapMsg);
    pointMapMsg.header.frame_id = "/body";
    pointMapMsg.header.stamp = ros::Time::now();

    pubPointCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Viewer/PointCloudMap", 100);
    pubMapPath = nh.advertise<nav_msgs::Path>("/Viewer/MapPath", 100);
    pubEigenMarker = nh.advertise<visualization_msgs::MarkerArray>("/Viewer/PD_Map", 100);

    while(ros::ok())
    {
        ros::Rate rate(0.1);
        pubPointCloudMap.publish(pointMapMsg);
        pubMapPath.publish(MapPath);
        pubEigenMarker.publish(eig_marker);
        rate.sleep();
    }
    return 0;
}
