// This is an advanced implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk


// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#define DEBUG_MODE_MAPPING 0

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

#include "lidarFactor.hpp"
#include "common.h"
#include "tic_toc.h"

#include "gd_loam/frame.h"

int frameCount = 0;
double ProcessTimeMean = 0;
int FrameNum = 0;

double timeLaserCloudCornerLast = 0;
double timeLaserCloudSurfLast = 0;
double timeLaserCloudFullRes = 0;
double timeLaserOdometry = 0;

bool initSequence = false;
// input: from odom
pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudCornerLastPD(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfLastPD(new pcl::PointCloud<PointType>());

std::vector<Eigen::Matrix3d> cornerCov;
std::vector<Eigen::Matrix3d> surfCov;
// ouput: all visualble cube points
pcl::PointCloud<PointType>::Ptr laserCloudSurround(new pcl::PointCloud<PointType>());

// surround points in map to build tree
pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap(new pcl::PointCloud<PointType>());

std::vector<Eigen::Matrix3d> cornerFromMapCov;
std::vector<Eigen::Matrix3d> surfFromMapCov;

//input & output: points in one frame. local --> global
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());

// every map points of probability distribution
pcl::PointCloud<PointType>::Ptr laserCloudCornerPDMap(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfPDMap(new pcl::PointCloud<PointType>());

std::vector<Eigen::Matrix3d> cornerMapCov;
std::vector<Eigen::Matrix3d> surfMapCov;

//Mapping result
pcl::PointCloud<PointType>::Ptr laserCloudMap(new pcl::PointCloud<PointType>());

//kd-tree
pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerMap(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfMap(new pcl::KdTreeFLANN<PointType>());

double parameters[7] = {0, 0, 0, 1, 0, 0, 0};
Eigen::Map<Eigen::Quaterniond> q_w_curr(parameters);
Eigen::Map<Eigen::Vector3d> t_w_curr(parameters + 4);

// wmap_T_odom * odom_T_curr = wmap_T_curr;
// transformation between odom's world and map's world frame
Eigen::Quaterniond q_wmap_wodom(1, 0, 0, 0);
Eigen::Vector3d t_wmap_wodom(0, 0, 0);

Eigen::Quaterniond q_wodom_curr(1, 0, 0, 0);
Eigen::Vector3d t_wodom_curr(0, 0, 0);


std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLastBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfLastBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullResBuf;
std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
std::mutex mBuf;

pcl::VoxelGrid<PointType> downSizeFilterMap;

std::vector<int> pointSearchInd;
std::vector<float> pointSearchSqDis;

PointType pointOri, pointSel;

ros::Publisher pubLaserCloudSurround, pubLaserCloudMap, pubLaserCloudFullRes, pubOdomAftMapped, pubOdomAftMappedHighFrec, pubLaserAfterMappedPath;
ros::Publisher pubLaserCloudFullResLocal;
ros::Publisher pubLaserCloudFeature;
ros::Publisher pubFrame;
ros::Publisher pubEigenMarker;
ros::Publisher pubProcessTime;

nav_msgs::Path laserAfterMappedPath;

int marker_id = 1;
visualization_msgs::MarkerArray eig_marker;

gd_loam::frame frameMsg;
pcl::PointCloud<PointType>::Ptr laserCloudFeatureLocal (new pcl::PointCloud<PointType> ());
pcl::PointCloud<PointType>::Ptr laserCloudFeatureMap (new pcl::PointCloud<PointType> ());

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

void clusteringCornerPointCloud(const pcl::PointCloud<PointType>::Ptr &corner, std::vector<int> clusterPicked,
                                std::vector<pcl::PointCloud<PointType>> &corner_clusterPC)
{
    pcl::PointCloud<PointType> cluster_points;
    for (size_t i = 0; i < corner->size() - 1; i++)
    {
        if (clusterPicked[i] == 1)  continue;

        cluster_points.clear();
        PointType t;
        t.x = corner->points[i].x;
        t.y = corner->points[i].y;
        t.z = corner->points[i].z;
        cluster_points.push_back(t);
        clusterPicked[i] = 1;
        float min_dist = 1;
        int currentInd = -1;
        for (size_t j = i + 1; j < corner->size(); j++)
        {
            PointType temp_curr_point = corner->points[i];
            PointType temp_comp_point = corner->points[j];
            float diffX = temp_curr_point.x - temp_comp_point.x;
            float diffY = temp_curr_point.y - temp_comp_point.y;
            float diffDist = sqrt(diffX * diffX + diffY * diffY);

            if (diffDist > min_dist)
            {
                PointType t;
                t.x = corner->points[currentInd].x;
                t.y = corner->points[currentInd].y;
                t.z = corner->points[currentInd].z;
                cluster_points.push_back(t);
            }
        }
        if (cluster_points.size() >= 5)
        {
            corner_clusterPC.push_back(cluster_points);
        }
    }
}

// set initial guess
void transformAssociateToMap()
{
    q_w_curr = q_wmap_wodom * q_wodom_curr;
    t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom;
}

void transformUpdate()
{
    q_wmap_wodom = q_w_curr * q_wodom_curr.inverse();
    t_wmap_wodom = t_w_curr - q_wmap_wodom * t_wodom_curr;
}

void pointAssociateToMap(PointType const *const pi, PointType *const po)
{
    Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
    Eigen::Vector3d point_w = q_w_curr * point_curr + t_w_curr;
    po->x = point_w.x();
    po->y = point_w.y();
    po->z = point_w.z();
    po->_PointXYZINormal::curvature = pi->_PointXYZINormal::curvature;
    po->_PointXYZINormal::normal_y = pi->_PointXYZINormal::normal_y;
    po->_PointXYZINormal::normal_z = pi->_PointXYZINormal::normal_z;

}

void laserCloudCornerLastHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudCornerLast2)
{
    mBuf.lock();
    cornerLastBuf.push(laserCloudCornerLast2);
    mBuf.unlock();
}

void laserCloudSurfLastHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudSurfLast2)
{
    mBuf.lock();
    surfLastBuf.push(laserCloudSurfLast2);
    mBuf.unlock();
}

void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
{
    mBuf.lock();
    fullResBuf.push(laserCloudFullRes2);
    mBuf.unlock();
}

void laserCloudFeatureHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFeature)
{
    laserCloudFeatureLocal->clear();
    pcl::fromROSMsg(*laserCloudFeature, *laserCloudFeatureLocal);
}

//receive odomtry
void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &laserOdometry)
{
    mBuf.lock();
    odometryBuf.push(laserOdometry);
    mBuf.unlock();

    // high frequence publish
    Eigen::Quaterniond q_wodom_curr;
    Eigen::Vector3d t_wodom_curr;
    q_wodom_curr.x() = laserOdometry->pose.pose.orientation.x;
    q_wodom_curr.y() = laserOdometry->pose.pose.orientation.y;
    q_wodom_curr.z() = laserOdometry->pose.pose.orientation.z;
    q_wodom_curr.w() = laserOdometry->pose.pose.orientation.w;
    t_wodom_curr.x() = laserOdometry->pose.pose.position.x;
    t_wodom_curr.y() = laserOdometry->pose.pose.position.y;
    t_wodom_curr.z() = laserOdometry->pose.pose.position.z;

    Eigen::Quaterniond q_w_curr = q_wmap_wodom * q_wodom_curr;
    Eigen::Vector3d t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom;

    nav_msgs::Odometry odomAftMapped;
    odomAftMapped.header.frame_id = "/body";
    odomAftMapped.child_frame_id = "/aft_mapped";
    odomAftMapped.header.stamp = laserOdometry->header.stamp;
    odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
    odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
    odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
    odomAftMapped.pose.pose.orientation.w = q_w_curr.w();
    odomAftMapped.pose.pose.position.x = t_w_curr.x();
    odomAftMapped.pose.pose.position.y = t_w_curr.y();
    odomAftMapped.pose.pose.position.z = t_w_curr.z();
    pubOdomAftMappedHighFrec.publish(odomAftMapped);
}

void process()
{
    while(1)
    {
        while (!cornerLastBuf.empty() && !surfLastBuf.empty() &&
            !fullResBuf.empty() && !odometryBuf.empty())
        {
            mBuf.lock();
            while (!odometryBuf.empty() && odometryBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
                odometryBuf.pop();
            if (odometryBuf.empty())
            {
                mBuf.unlock();
                break;
            }

            while (!surfLastBuf.empty() && surfLastBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
                surfLastBuf.pop();
            if (surfLastBuf.empty())
            {
                mBuf.unlock();
                break;
            }

            while (!fullResBuf.empty() && fullResBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
                fullResBuf.pop();
            if (fullResBuf.empty())
            {
                mBuf.unlock();
                break;
            }

            timeLaserCloudCornerLast = cornerLastBuf.front()->header.stamp.toSec();
            timeLaserCloudSurfLast = surfLastBuf.front()->header.stamp.toSec();
            timeLaserCloudFullRes = fullResBuf.front()->header.stamp.toSec();
            timeLaserOdometry = odometryBuf.front()->header.stamp.toSec();
            ros::Time OdometryTime = odometryBuf.front()->header.stamp;

            if (timeLaserCloudCornerLast != timeLaserOdometry ||
                timeLaserCloudSurfLast != timeLaserOdometry ||
                timeLaserCloudFullRes != timeLaserOdometry)
            {
                //printf("time corner %f surf %f full %f odom %f \n", timeLaserCloudCornerLast, timeLaserCloudSurfLast, timeLaserCloudFullRes, timeLaserOdometry);
                //printf("unsync messeage!");
                mBuf.unlock();
                break;
            }


            laserCloudCornerLast->clear();

            pcl::fromROSMsg(*cornerLastBuf.front(), *laserCloudCornerLast);

            cornerLastBuf.pop();

            laserCloudSurfLast->clear();

            pcl::fromROSMsg(*surfLastBuf.front(), *laserCloudSurfLast);


            surfLastBuf.pop();

            laserCloudFullRes->clear();
            pcl::fromROSMsg(*fullResBuf.front(), *laserCloudFullRes);
            fullResBuf.pop();

            q_wodom_curr.x() = odometryBuf.front()->pose.pose.orientation.x;
            q_wodom_curr.y() = odometryBuf.front()->pose.pose.orientation.y;
            q_wodom_curr.z() = odometryBuf.front()->pose.pose.orientation.z;
            q_wodom_curr.w() = odometryBuf.front()->pose.pose.orientation.w;
            t_wodom_curr.x() = odometryBuf.front()->pose.pose.position.x;
            t_wodom_curr.y() = odometryBuf.front()->pose.pose.position.y;
            t_wodom_curr.z() = odometryBuf.front()->pose.pose.position.z;
            odometryBuf.pop();

            while(!cornerLastBuf.empty())
            {
                cornerLastBuf.pop();
                //printf("drop lidar frame in mapping for real time performance \n");
            }

            mBuf.unlock();

            TicToc t_whole;

            transformAssociateToMap();

            TicToc t_shift;

            laserCloudCornerFromMap->clear();
            laserCloudSurfFromMap->clear();

            cornerFromMapCov.clear();
            surfFromMapCov.clear();


            if (laserCloudCornerPDMap->points.size() > 0)
            {
                for (size_t ss = 0; ss < laserCloudCornerPDMap->size(); ss++)
                {
                    Eigen::Vector3d diffVec;
                    diffVec(0) = laserCloudCornerPDMap->points[ss].x - t_w_curr.x();
                    diffVec(1) = laserCloudCornerPDMap->points[ss].y - t_w_curr.y();
                    diffVec(2) = laserCloudCornerPDMap->points[ss].z - t_w_curr.z();

                    double diffDist = sqrt(pow(diffVec(0),2)+pow(diffVec(1),2)+pow(diffVec(0),2));
                    if (diffDist < 50)
                    {
                        laserCloudCornerFromMap->points.push_back(laserCloudCornerPDMap->points[ss]);
                        cornerFromMapCov.push_back(cornerMapCov[ss]);
                    }
                }
            }
            if (laserCloudSurfPDMap->points.size() > 0)
            {
                for (size_t ss = 0; ss < laserCloudSurfPDMap->size(); ss++)
                {
                    Eigen::Vector3d diffVec;
                    diffVec(0) = laserCloudSurfPDMap->points[ss].x - t_w_curr.x();
                    diffVec(1) = laserCloudSurfPDMap->points[ss].y - t_w_curr.y();
                    diffVec(2) = laserCloudSurfPDMap->points[ss].z - t_w_curr.z();

                    double diffDist = sqrt(pow(diffVec(0),2)+pow(diffVec(1),2)+pow(diffVec(0),2));
                    if (diffDist < 50)
                    {
                        laserCloudSurfFromMap->points.push_back(laserCloudSurfPDMap->points[ss]);
                        surfFromMapCov.push_back(surfMapCov[ss]);
                    }
                }
            }

            int laserCloudCornerFromMapNum = laserCloudCornerFromMap->points.size();
            int laserCloudSurfFromMapNum = laserCloudSurfFromMap->points.size();

            pcl::PointCloud<PointType>::Ptr laserCloudCornerStack(new pcl::PointCloud<PointType>());
            *laserCloudCornerStack = *laserCloudCornerLast;
            int laserCloudCornerStackNum = laserCloudCornerStack->points.size();

            pcl::PointCloud<PointType>::Ptr laserCloudSurfStack(new pcl::PointCloud<PointType>());
            *laserCloudSurfStack = *laserCloudSurfLast;
            int laserCloudSurfStackNum = laserCloudSurfStack->points.size();

            //printf("map prepare time %f ms\n", t_shift.toc());
            //printf("last corner num %d  surf num %d \n", laserCloudCornerStackNum, laserCloudSurfStackNum);
            TicToc t_opt;
            TicToc t_tree;
            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMap);
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMap);
            //printf("build tree time %f ms \n", t_tree.toc());

            for (int iterCount = 0; iterCount < 2; iterCount++)
            {
                //ceres::LossFunction *loss_function = NULL;
                ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                ceres::LocalParameterization *q_parameterization =
                    new ceres::EigenQuaternionParameterization();
                ceres::Problem::Options problem_options;

                ceres::Problem problem(problem_options);
                problem.AddParameterBlock(parameters, 4, q_parameterization);
                problem.AddParameterBlock(parameters + 4, 3);

                TicToc t_data;
                int corner_num = 0;
                if (laserCloudCornerFromMapNum > 0)
                {
                    for (int i = 0; i < laserCloudCornerStackNum; i++)
                    {
                        pointOri = laserCloudCornerStack->points[i];
                        //double sqrtDis = pointOri.x * pointOri.x + pointOri.y * pointOri.y + pointOri.z * pointOri.z;
                        pointAssociateToMap(&pointOri, &pointSel);
                        kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);
                        double minPointSqDis = 3;
                        int minPointInd = -1;
                        for (size_t s = 0; s < pointSearchInd.size(); s++)
                        {
                            if (pointSearchSqDis[s] < 5)
                            {
                                double pointSqDis = (laserCloudCornerFromMap->points[pointSearchInd[s]].x - pointSel.x) *
                                                    (laserCloudCornerFromMap->points[pointSearchInd[s]].x - pointSel.x) +
                                                    (laserCloudCornerFromMap->points[pointSearchInd[s]].y - pointSel.y) *
                                                    (laserCloudCornerFromMap->points[pointSearchInd[s]].y - pointSel.y) +
                                                    (laserCloudCornerFromMap->points[pointSearchInd[s]].z - pointSel.z) *
                                                    (laserCloudCornerFromMap->points[pointSearchInd[s]].z - pointSel.z);

                                if (pointSqDis < minPointSqDis)
                                {
                                    // find nearer point
                                    minPointSqDis = pointSqDis;
                                    minPointInd = pointSearchInd[s];
                                }
                            }
                        }
                        if (minPointInd >= 0)
                        {
                            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cornerFromMapCov[minPointInd]);

                            // if is indeed line feature
                            // note Eigen library sort eigenvalues in increasing order
                            Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);

                            Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
                            if (saes.eigenvalues()[2] > 16 * saes.eigenvalues()[1])
                            {
                                Eigen::Vector3d point_on_line;
                                point_on_line(0) = laserCloudCornerFromMap->points[minPointInd].x;
                                point_on_line(1) = laserCloudCornerFromMap->points[minPointInd].y;
                                point_on_line(2) = laserCloudCornerFromMap->points[minPointInd].z;
                                Eigen::Vector3d point_a, point_b;
                                point_a = 0.1 * unit_direction + point_on_line;
                                point_b = -0.1 * unit_direction + point_on_line;
                                ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, point_a, point_b, 1.0);
                                problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
                                corner_num++;
                            }
                        }
                    }
                }
                else
                {
                    ROS_WARN("time Map corner num are not enough");
                }

                int surf_num = 0;
                if (laserCloudSurfFromMapNum > 50)
                {
                    for (int i = 0; i < laserCloudSurfStackNum; i++)
                    {
                        pointOri = laserCloudSurfStack->points[i];
                        //double sqrtDis = pointOri.x * pointOri.x + pointOri.y * pointOri.y + pointOri.z * pointOri.z;
                        pointAssociateToMap(&pointOri, &pointSel);
                        kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

                        double minPointSqDis = 2;
                        int minPointInd = -1;
                        for (size_t s = 0; s < pointSearchInd.size(); s++)
                        {
                            if (pointSearchSqDis[s] < 4)
                            {
                                double pointSqDis = (laserCloudSurfFromMap->points[pointSearchInd[s]].x - pointSel.x) *
                                                    (laserCloudSurfFromMap->points[pointSearchInd[s]].x - pointSel.x) +
                                                    (laserCloudSurfFromMap->points[pointSearchInd[s]].y - pointSel.y) *
                                                    (laserCloudSurfFromMap->points[pointSearchInd[s]].y - pointSel.y) +
                                                    (laserCloudSurfFromMap->points[pointSearchInd[s]].z - pointSel.z) *
                                                    (laserCloudSurfFromMap->points[pointSearchInd[s]].z - pointSel.z);

                                if (pointSqDis < minPointSqDis)
                                {
                                    // find nearer point
                                    minPointSqDis = pointSqDis;
                                    minPointInd = pointSearchInd[s];
                                }
                            }
                        }
                        if (minPointInd >= 0)
                        {
                            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(surfFromMapCov[minPointInd]);
                            if (saes.eigenvalues()[0] < pow(0.05,2))
                            {
                                // if is indeed line feature
                                // note Eigen library sort eigenvalues in increasing order
                                Eigen::Vector3d direction1 = saes.eigenvectors().col(2);
                                Eigen::Vector3d direction2 = saes.eigenvectors().col(1);

                                Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);

                                Eigen::Vector3d point_on_surf;
                                point_on_surf(0) = laserCloudSurfFromMap->points[minPointInd].x;
                                point_on_surf(1) = laserCloudSurfFromMap->points[minPointInd].y;
                                point_on_surf(2) = laserCloudSurfFromMap->points[minPointInd].z;
                                Eigen::Vector3d point_a, point_b, point_c;
                                point_a = 0.1 * direction1 + point_on_surf;
                                point_b = -0.1 * direction1 + point_on_surf;
                                point_c = 0.1 * direction2 + point_on_surf;

                                ceres::CostFunction *cost_function = LidarPlaneFactor::Create(curr_point, point_a, point_b, point_c, 1.0);
                                problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
                                surf_num++;
                            }
                        }
                    }
                }
                else
                {
                    ROS_WARN("time Map surf num are not enough");
                }
                //printf("mapping data assosiation time %f ms \n", t_data.toc());
                if (corner_num + surf_num > 50)
                {
                    TicToc t_solver;
                    ceres::Solver::Options options;
                    options.linear_solver_type = ceres::DENSE_QR;
                    options.max_num_iterations = 4;
                    options.minimizer_progress_to_stdout = false;
                    options.check_gradients = false;
                    options.gradient_check_relative_precision = 1e-4;
                    ceres::Solver::Summary summary;
                    ceres::Solve(options, &problem, &summary);
                    //printf("mapping solver time %f ms \n", t_solver.toc());

                    ////printf("time %f \n", timeLaserOdometry);
                    //printf("corner factor num %d surf factor num %d\n", corner_num, surf_num);
                }
            }
            //printf("mapping optimization time %f \n", t_opt.toc());

            transformUpdate();

            TicToc t_add;
            pcl::PointCloud<PointType>::Ptr cornerTemp (new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr surfTemp (new pcl::PointCloud<PointType>());
            for (int i = 0; i < laserCloudCornerStackNum; i++)
            {
                pointAssociateToMap(&laserCloudCornerStack->points[i], &pointSel);
                cornerTemp->push_back(pointSel);
            }

            for (int i = 0; i < laserCloudSurfStackNum; i++)
            {
                pointAssociateToMap(&laserCloudSurfStack->points[i], &pointSel);
                surfTemp->push_back(pointSel);
            }

            laserCloudCornerLastPD->clear();
            cornerCov.clear();
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
                        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cov);
                        if (saes.eigenvalues()[2] > 16*saes.eigenvalues()[1] && saes.eigenvalues()[1] < pow(0.2,2))
                        {
                            PointType tempPoint;
                            tempPoint.x = mean(0);
                            tempPoint.y = mean(1);
                            tempPoint.z = mean(2);
                            laserCloudCornerLastPD->push_back(tempPoint);
                            cornerCov.push_back(cov);
                        }

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
                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cov);
                    if (saes.eigenvalues()[2] > 16*saes.eigenvalues()[1] && saes.eigenvalues()[1] < pow(0.2,2))
                    {
                        PointType tempPoint;
                        tempPoint.x = mean(0);
                        tempPoint.y = mean(1);
                        tempPoint.z = mean(2);
                        laserCloudCornerLastPD->push_back(tempPoint);
                        cornerCov.push_back(cov);
                    }

                    tempPC.clear();
                    tempPC.push_back(cornerTemp->points[k]);
                    cnt++;
                }
            }

            laserCloudSurfLastPD->clear();
            surfCov.clear();
            cnt = 0;

            for (size_t k = 0; k < surfTemp->points.size(); k++)
            {
                if (surfTemp->points[k]._PointXYZINormal::normal_y == cnt)
                {
                    tempPC.push_back(surfTemp->points[k]);

                    if (k == surfTemp->points.size()-1)
                    {
                        Eigen::Vector3d mean = getMean(tempPC);
                        Eigen::Matrix3d cov = getCovariance(tempPC);

                        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cov);
                        if (saes.eigenvalues()[0] < pow(0.05,2))
                        {
                            PointType tempPoint;
                            tempPoint.x = mean(0);
                            tempPoint.y = mean(1);
                            tempPoint.z = mean(2);
                            laserCloudSurfLastPD->push_back(tempPoint);
                            surfCov.push_back(cov);
                        }

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
                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cov);
                    if (saes.eigenvalues()[0] < pow(0.05,2))
                    {
                        laserCloudSurfLastPD->push_back(tempPoint);
                        surfCov.push_back(cov);
                    }
                    tempPC.clear();
                    tempPC.push_back(surfTemp->points[k]);
                    cnt++;
                }
            }

            if (initSequence == false)
            {
                int laserCloudCornerLastPDNum = laserCloudCornerLastPD->points.size();
                int laserCloudSurfLastPDNum = laserCloudSurfLastPD->points.size();
                if (laserCloudCornerLastPDNum > 0 && laserCloudSurfLastPDNum > 0)
                {
                    *laserCloudCornerPDMap = *laserCloudCornerLastPD;
                    *laserCloudSurfPDMap = *laserCloudSurfLastPD;

                    cornerMapCov = cornerCov;
                    surfMapCov = surfCov;
                    initSequence = true;
                }
            }
            else
            {
                kdtreeCornerMap->setInputCloud(laserCloudCornerPDMap);
                kdtreeSurfMap->setInputCloud(laserCloudSurfPDMap);

                std::vector<int> MapSearchInd;
                std::vector<float> MapSearchSqDis;
                for(size_t c = 0; c < laserCloudCornerLastPD->points.size(); c++)
                {
                     kdtreeCornerMap->nearestKSearch(laserCloudCornerLastPD->points[c], 1, MapSearchInd, MapSearchSqDis);
                     if (MapSearchSqDis[0] > 3)
                     {
                         Eigen::Vector3d diff_mean;
                         diff_mean(0) = laserCloudCornerLastPD->points[c].x - laserCloudCornerPDMap->points[MapSearchInd[0]].x;
                         diff_mean(1) = laserCloudCornerLastPD->points[c].y - laserCloudCornerPDMap->points[MapSearchInd[0]].y;
                         diff_mean(2) = laserCloudCornerLastPD->points[c].z - laserCloudCornerPDMap->points[MapSearchInd[0]].z;
                         Eigen::Matrix3d mapCov = cornerMapCov[MapSearchInd[0]];
                         double maha_dist;
                         maha_dist = sqrt(diff_mean.transpose() * mapCov.inverse() * diff_mean);
                         if (maha_dist > 5)
                         {
                            laserCloudCornerPDMap->points.push_back(laserCloudCornerLastPD->points[c]);
                            cornerMapCov.push_back(cornerCov[c]);
                         }
                    }
                }

                for(size_t c = 0; c < laserCloudSurfLastPD->points.size(); c++)
                {
                    kdtreeSurfMap->nearestKSearch(laserCloudSurfLastPD->points[c], 1, MapSearchInd, MapSearchSqDis);
                    if (MapSearchSqDis[0] > 3)
                    {
                        Eigen::Vector3d diff_mean;
                        diff_mean(0) = laserCloudSurfLastPD->points[c].x - laserCloudSurfPDMap->points[MapSearchInd[0]].x;
                        diff_mean(1) = laserCloudSurfLastPD->points[c].y - laserCloudSurfPDMap->points[MapSearchInd[0]].y;
                        diff_mean(2) = laserCloudSurfLastPD->points[c].z - laserCloudSurfPDMap->points[MapSearchInd[0]].z;
                        Eigen::Matrix3d mapCov = surfMapCov[MapSearchInd[0]];
                        double maha_dist;
                        maha_dist = diff_mean.transpose() * mapCov.inverse() * diff_mean;
                        if (maha_dist > 5)
                        {
                            laserCloudSurfPDMap->points.push_back(laserCloudSurfLastPD->points[c]);
                            surfMapCov.push_back(surfCov[c]);
                        }
                    }
                }
            }

#if DEBUG_MODE_MAPPING == 1
            for (size_t c = 0; c < cornerMapCov.size(); c++)
            {
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cornerMapCov[c]);
                Eigen::Vector3d mean;
                mean(0) = laserCloudCornerPDMap->points[c].x;
                mean(1) = laserCloudCornerPDMap->points[c].y;
                mean(2) = laserCloudCornerPDMap->points[c].z;
                drawEigenVector(OdometryTime, 1, marker_id, mean, saes.eigenvectors(), saes.eigenvalues(), eig_marker);
                marker_id = eig_marker.markers.back().id+1;
            }

            for (size_t s = 0; s < surfMapCov.size(); s++)
            {
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(surfMapCov[s]);
                Eigen::Vector3d mean;
                mean(0) = laserCloudSurfPDMap->points[s].x;
                mean(1) = laserCloudSurfPDMap->points[s].y;
                mean(2) = laserCloudSurfPDMap->points[s].z;
                drawEigenVector(OdometryTime, 2, marker_id, mean, saes.eigenvectors(), saes.eigenvalues(), eig_marker);
                marker_id = eig_marker.markers.back().id+1;
            }
#endif
            //printf("filter time %f ms \n", t_filter.toc());

            TicToc t_pub;
            //publish surround map for every 5 frame

            sensor_msgs::PointCloud2 laserCloudFullRes3Local;
            pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3Local);
            laserCloudFullRes3Local.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            laserCloudFullRes3Local.header.frame_id = "/body";
            pubLaserCloudFullResLocal.publish(laserCloudFullRes3Local);

            int laserCloudFullResNum = laserCloudFullRes->points.size();
            for (int i = 0; i < laserCloudFullResNum; i++)
            {
                pointAssociateToMap(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]);
            }

            sensor_msgs::PointCloud2 laserCloudFullRes3;
            pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
            laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            laserCloudFullRes3.header.frame_id = "/body";
            pubLaserCloudFullRes.publish(laserCloudFullRes3);

            laserCloudFeatureMap->clear();
            for (size_t i = 0; i < laserCloudFeatureLocal->size(); i++)
            {
                PointType pointP;
                pointAssociateToMap(&laserCloudFeatureLocal->points[i], &pointP);
                laserCloudFeatureMap->points.push_back(pointP);
            }
            sensor_msgs::PointCloud2 laserCloudFeature2;
            pcl::toROSMsg(*laserCloudFeatureMap, laserCloudFeature2);
            laserCloudFeature2.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            laserCloudFeature2.header.frame_id = "/body";
            pubLaserCloudFeature.publish(laserCloudFeature2);

            if (frameCount % 20 == 0)
            {
                *laserCloudMap += *laserCloudFullRes;

                downSizeFilterMap.setInputCloud(laserCloudMap);
                downSizeFilterMap.filter(*laserCloudMap);


                sensor_msgs::PointCloud2 laserCloudMsg;
                pcl::toROSMsg(*laserCloudMap, laserCloudMsg);
                laserCloudMsg.header.stamp = ros::Time().fromSec(timeLaserOdometry);
                laserCloudMsg.header.frame_id = "/body";
                pubLaserCloudMap.publish(laserCloudMsg);
            }

            if (frameCount % 5 == 0)
            {
                laserCloudSurround->clear();
                pcl::ConditionAnd<PointType>::Ptr rangeCondition (new pcl::ConditionAnd<PointType> ());
                pcl::PointCloud<PointType>::Ptr filtered_map_ptr (new pcl::PointCloud<PointType>());
                rangeCondition->addComparison (pcl::FieldComparison<PointType>::ConstPtr (new pcl::FieldComparison<PointType> ("x", pcl::ComparisonOps::GT, t_w_curr.x() -50)));
                rangeCondition->addComparison (pcl::FieldComparison<PointType>::ConstPtr (new pcl::FieldComparison<PointType> ("x", pcl::ComparisonOps::LT, t_w_curr.x() +50)));
                rangeCondition->addComparison (pcl::FieldComparison<PointType>::ConstPtr (new pcl::FieldComparison<PointType> ("y", pcl::ComparisonOps::GT, t_w_curr.y() -50)));
                rangeCondition->addComparison (pcl::FieldComparison<PointType>::ConstPtr (new pcl::FieldComparison<PointType> ("y", pcl::ComparisonOps::LT, t_w_curr.y() +50)));
                pcl::ConditionalRemoval<PointType> ConditionRemoval;
                ConditionRemoval.setCondition (rangeCondition);
                ConditionRemoval.setInputCloud (laserCloudMap);
                ConditionRemoval.filter (*laserCloudSurround);

                sensor_msgs::PointCloud2 laserCloudSurround3;
                pcl::toROSMsg(*laserCloudSurround, laserCloudSurround3);
                laserCloudSurround3.header.stamp = ros::Time().fromSec(timeLaserOdometry);
                laserCloudSurround3.header.frame_id = "/body";
                pubLaserCloudSurround.publish(laserCloudSurround3);
            }

            //printf("mapping pub time %f ms \n", t_pub.toc());

            //printf("whole mapping time %f ms ++++++++++\n", t_whole.toc());


            double wholeProcessTime = t_whole.toc();
            ProcessTimeMean = ProcessTimeMean*FrameNum + wholeProcessTime;
            FrameNum++;
            ProcessTimeMean /= FrameNum;

            std_msgs::Float32 float_time;
            float_time.data = (float)ProcessTimeMean;
            pubProcessTime.publish(float_time);

            nav_msgs::Odometry odomAftMapped;
            odomAftMapped.header.frame_id = "/body";
            odomAftMapped.child_frame_id = "/aft_mapped";
            odomAftMapped.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
            odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
            odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
            odomAftMapped.pose.pose.orientation.w = q_w_curr.w();
            odomAftMapped.pose.pose.position.x = t_w_curr.x();
            odomAftMapped.pose.pose.position.y = t_w_curr.y();
            odomAftMapped.pose.pose.position.z = t_w_curr.z();
            pubOdomAftMapped.publish(odomAftMapped);

            frameMsg.header.frame_id = "/body";
            frameMsg.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            frameMsg.fullPC = laserCloudFullRes3Local;
            sensor_msgs::PointCloud2 laserCornerMsg;
            pcl::toROSMsg(*laserCloudCornerLast, laserCornerMsg);
            laserCornerMsg.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            laserCornerMsg.header.frame_id = "/body";
            frameMsg.CornerPC = laserCornerMsg;
            sensor_msgs::PointCloud2 laserSurfMsg;
            pcl::toROSMsg(*laserCloudSurfLast, laserSurfMsg);
            laserSurfMsg.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            laserSurfMsg.header.frame_id = "/body";
            frameMsg.SurfPC = laserSurfMsg;
            frameMsg.pose = odomAftMapped;
            frameMsg.frame_idx = frameCount;
            pubFrame.publish(frameMsg);

            geometry_msgs::PoseStamped laserAfterMappedPose;
            laserAfterMappedPose.header = odomAftMapped.header;
            laserAfterMappedPose.pose = odomAftMapped.pose.pose;

            laserAfterMappedPath.header.stamp = odomAftMapped.header.stamp;
            laserAfterMappedPath.header.frame_id = "/body";
            laserAfterMappedPath.poses.push_back(laserAfterMappedPose);
            pubLaserAfterMappedPath.publish(laserAfterMappedPath);

            static tf::TransformBroadcaster br;
            tf::Transform transform;
            tf::Quaternion q;
            transform.setOrigin(tf::Vector3(t_w_curr(0),
                                            t_w_curr(1),
                                            t_w_curr(2)));
            q.setW(q_w_curr.w());
            q.setX(q_w_curr.x());
            q.setY(q_w_curr.y());
            q.setZ(q_w_curr.z());
            transform.setRotation(q);
            br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "/body", "/aft_mapped"));

#if DEBUG_MODE_MAPPING == 1
            pubEigenMarker.publish(eig_marker);
            eig_marker.markers.clear();
            marker_id = 1;
#endif


            frameCount++;
        }
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    float mapRes = 0;
    nh.param<float>("mapping_resolution", mapRes, 0.5);
    downSizeFilterMap.setLeafSize(mapRes, mapRes,mapRes);

    ros::Subscriber subLaserCloudCornerLast = nh.subscribe<sensor_msgs::PointCloud2>("/odometry/laser_cloud_corner_last", 100, laserCloudCornerLastHandler);

    ros::Subscriber subLaserCloudSurfLast = nh.subscribe<sensor_msgs::PointCloud2>("/odometry/laser_cloud_surf_last", 100, laserCloudSurfLastHandler);

    ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/odometry/laser_odom_to_init", 100, laserOdometryHandler);

    ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100, laserCloudFullResHandler);

    ros::Subscriber subLaserCloudFeature = nh.subscribe<sensor_msgs::PointCloud2>("/feature/laser_feature_cluster_points", 100, laserCloudFeatureHandler);

    pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("/mapping/laser_cloud_surround", 100);

    pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/mapping/laser_cloud_map", 100);

    pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/mapping/velodyne_cloud_registered", 100);
    pubLaserCloudFullResLocal = nh.advertise<sensor_msgs::PointCloud2>("/mapping/velodyne_cloud_registered_local", 100);

    pubLaserCloudFeature = nh.advertise<sensor_msgs::PointCloud2>("/mapping/aft_mapped_laser_feature_cluster_points", 100);

    pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/mapping/aft_mapped_to_init", 100);

    pubOdomAftMappedHighFrec = nh.advertise<nav_msgs::Odometry>("/mapping/aft_mapped_to_init_high_frec", 100);

    pubLaserAfterMappedPath = nh.advertise<nav_msgs::Path>("/mapping/aft_mapped_path", 100);

    pubEigenMarker = nh.advertise<visualization_msgs::MarkerArray>("/map_eigen", 100);

    pubFrame = nh.advertise<gd_loam::frame>("/mapping/frame", 100);

    pubProcessTime = nh.advertise<std_msgs::Float32>("/mapping_process_time", 100);

    std::thread mapping_process{process};

    ros::spin();

    return 0;
}
