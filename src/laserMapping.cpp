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

#include <fstream>
#include <math.h>
#include <vector>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <string>
#include <optional>
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

#include "lidarFactor.hpp"
#include "common.h"
#include "tic_toc.h"

#include "pd_loam/frame.h"
#include "pd_loam/gnss.h"

int frameCount = 0;
double ProcessTimeMean = 0;
int FrameNum = 0;

double timeLaserCloudCornerLast = 0;
double timeLaserCloudSurfLast = 0;
double timeLaserCloudFullRes = 0;
double timeLaserOdometry = 0;
double timeGNSS = 0;

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

//kd-tree
pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerMap(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfMap(new pcl::KdTreeFLANN<PointType>());

pd_loam::gnss::ConstPtr gnssMsg;

double parameters[7] = {0, 0, 0, 0, 0, 0, 1};
Eigen::Map<Eigen::Quaterniond> q_w_curr(parameters + 3);
Eigen::Map<Eigen::Vector3d> t_w_curr(parameters);

// wmap_T_odom * odom_T_curr = wmap_T_curr;
// transformation between odom's world and map's world frame
Eigen::Quaterniond q_wmap_wodom(1, 0, 0, 0);
Eigen::Vector3d t_wmap_wodom(0, 0, 0);

Eigen::Quaterniond q_wodom_curr(1, 0, 0, 0);
Eigen::Vector3d t_wodom_curr(0, 0, 0);

float mapMatchingRange = 0;
double matchingThres = 0;

std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLastBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfLastBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullResBuf;
std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
std::queue<pd_loam::gnss::ConstPtr> gnssBuf;
std::mutex mBuf;
std::mutex mMapping;

std::vector<int> pointSearchInd;
std::vector<float> pointSearchSqDis;

PointType pointOri, pointSel;

ros::Publisher pubLaserCloudFullRes, pubOdomAftMapped;
ros::Publisher pubLaserCloudFullResLocal;
ros::Publisher pubLaserCloudFeature;
ros::Publisher pubLaserAfterMappedPath, pubGnssPath, pubKalmanPath;
ros::Publisher pubFrame;
ros::Publisher pubEigenMarker;
ros::Publisher pubProcessTime, pubCost;

nav_msgs::Path laserAfterMappedPath, gnssEnuPath, KalmanPath;

int marker_id = 1;
visualization_msgs::MarkerArray eig_marker;

pd_loam::frame frameMsg;
pcl::PointCloud<PointType>::Ptr laserCloudFeatureLocal (new pcl::PointCloud<PointType> ());
pcl::PointCloud<PointType>::Ptr laserCloudFeatureMap (new pcl::PointCloud<PointType> ());

double final_cost = 100;
Eigen::VectorXd state(6);
Eigen::MatrixXd Q(6,6);
Eigen::MatrixXd H(9,6);
Eigen::MatrixXd P(6,6);
Eigen::MatrixXd R(9,9);
Eigen::MatrixXd K(6,9);

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
                laserCloudCornerLastPD->push_back(tempPoint);
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
            laserCloudCornerLastPD->push_back(tempPoint);
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
                laserCloudSurfLastPD->push_back(tempPoint);
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

            laserCloudSurfLastPD->push_back(tempPoint);
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
    if (laserCloudCornerPDMap->points.size() > 0)
    {
        kdtreeCornerMap->setInputCloud(laserCloudCornerPDMap);
        for(size_t c = 0; c < laserCloudCornerLastPD->points.size(); c++)
        {
             kdtreeCornerMap->nearestKSearch(laserCloudCornerLastPD->points[c], 1, MapSearchInd, MapSearchSqDis);
             Eigen::Vector3d diff_mean;
             diff_mean(0) = laserCloudCornerLastPD->points[c].x - laserCloudCornerPDMap->points[MapSearchInd[0]].x;
             diff_mean(1) = laserCloudCornerLastPD->points[c].y - laserCloudCornerPDMap->points[MapSearchInd[0]].y;
             diff_mean(2) = laserCloudCornerLastPD->points[c].z - laserCloudCornerPDMap->points[MapSearchInd[0]].z;
             Eigen::Matrix3d mapCov = cornerMapCov[MapSearchInd[0]];
             double maha_dist = sqrt(diff_mean.transpose() * mapCov.inverse() * diff_mean);
             if (MapSearchSqDis[0] > matchingThres && maha_dist > 5)
             {
                laserCloudCornerPDMap->points.push_back(laserCloudCornerLastPD->points[c]);
                cornerMapCov.push_back(cornerCov[c]);
            }
        }
    }
    else
    {
        *laserCloudCornerPDMap = *laserCloudCornerLastPD;
        cornerMapCov = cornerCov;
    }
    //erase old corner PD map
    while(laserCloudCornerPDMap->points.size() > 1000)
    {
        laserCloudCornerPDMap->points.erase(laserCloudCornerPDMap->points.begin());
        cornerMapCov.erase(cornerMapCov.begin());
    }

    if (laserCloudSurfPDMap->points.size() > 0)
    {
        kdtreeSurfMap->setInputCloud(laserCloudSurfPDMap);
        for(size_t c = 0; c < laserCloudSurfLastPD->points.size(); c++)
        {
            kdtreeSurfMap->nearestKSearch(laserCloudSurfLastPD->points[c], 1, MapSearchInd, MapSearchSqDis);
            Eigen::Vector3d diff_mean;
            diff_mean(0) = laserCloudSurfLastPD->points[c].x - laserCloudSurfPDMap->points[MapSearchInd[0]].x;
            diff_mean(1) = laserCloudSurfLastPD->points[c].y - laserCloudSurfPDMap->points[MapSearchInd[0]].y;
            diff_mean(2) = laserCloudSurfLastPD->points[c].z - laserCloudSurfPDMap->points[MapSearchInd[0]].z;
            Eigen::Matrix3d mapCov = surfMapCov[MapSearchInd[0]];
            double maha_dist = diff_mean.transpose() * mapCov.inverse() * diff_mean;
            if (MapSearchSqDis[0] > matchingThres && maha_dist > 5)
            {
                laserCloudSurfPDMap->points.push_back(laserCloudSurfLastPD->points[c]);
                surfMapCov.push_back(surfCov[c]);
            }
        }
    }
    else
    {
        *laserCloudSurfPDMap = *laserCloudSurfLastPD;
        surfMapCov = surfCov;
    }
    //erase old surf PD map
    while(laserCloudSurfPDMap->points.size() > 10000)
    {
        laserCloudSurfPDMap->points.erase(laserCloudSurfPDMap->points.begin());
        surfMapCov.erase(surfMapCov.begin());
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
    po->intensity = pi->intensity;
    po->_PointXYZINormal::curvature = pi->_PointXYZINormal::curvature;
    po->_PointXYZINormal::normal_y = pi->_PointXYZINormal::normal_y;
    po->_PointXYZINormal::normal_z = pi->_PointXYZINormal::normal_z;

}

void gnssHandler(const pd_loam::gnssConstPtr &gnss)
{
    mBuf.lock();
    gnssBuf.push(gnss);
    mBuf.unlock();
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
}

void process()
{
    while(1)
    {
        while (!cornerLastBuf.empty() && !surfLastBuf.empty() &&
            !fullResBuf.empty() && !odometryBuf.empty() && !gnssBuf.empty())
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

            while (!gnssBuf.empty() && gnssBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
                gnssBuf.pop();
            if (gnssBuf.empty())
            {
                mBuf.unlock();
                break;
            }

            timeLaserCloudCornerLast = cornerLastBuf.front()->header.stamp.toSec();
            timeLaserCloudSurfLast = surfLastBuf.front()->header.stamp.toSec();
            timeLaserCloudFullRes = fullResBuf.front()->header.stamp.toSec();
            timeLaserOdometry = odometryBuf.front()->header.stamp.toSec();
            timeGNSS = gnssBuf.front()->header.stamp.toSec();

            if (timeLaserCloudCornerLast != timeLaserOdometry ||
                timeLaserCloudSurfLast != timeLaserOdometry ||
                timeLaserCloudFullRes != timeLaserOdometry ||
                timeGNSS != timeLaserOdometry)
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
            pcl::VoxelGrid<PointType> downSizeFilter;
            downSizeFilter.setInputCloud(laserCloudFullRes);
            downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
            downSizeFilter.filter(*laserCloudFullRes);
            fullResBuf.pop();

            gnssMsg = gnssBuf.front();
            gnssBuf.pop();

            tf::Quaternion q_Inc;
            q_Inc.setRPY(deg2rad(gnssMsg->roll_inc), deg2rad(gnssMsg->pitch_inc), deg2rad(gnssMsg->azi_inc));
            q_wodom_curr.w() = q_Inc.w();
            q_wodom_curr.x() = q_Inc.x();
            q_wodom_curr.y() = q_Inc.y();
            q_wodom_curr.z() = q_Inc.z();
            t_wodom_curr.x() = gnssMsg->x_vel * gnssMsg->dt;
            t_wodom_curr.y() = gnssMsg->y_vel * gnssMsg->dt;
            t_wodom_curr.z() = gnssMsg->z_vel * gnssMsg->dt;

            odometryBuf.pop();

            while(!cornerLastBuf.empty())
            {
                cornerLastBuf.pop();
                //printf("drop lidar frame in mapping for real time performance \n");
            }

            mBuf.unlock();

            TicToc t_whole;

            mMapping.lock();
            transformAssociateToMap();

            pcl::PointCloud<PointType>::Ptr laserCloudCornerStack(new pcl::PointCloud<PointType>());
            *laserCloudCornerStack = *laserCloudCornerLast;
            int laserCloudCornerStackNum = laserCloudCornerStack->points.size();

            pcl::PointCloud<PointType>::Ptr laserCloudSurfStack(new pcl::PointCloud<PointType>());
            *laserCloudSurfStack = *laserCloudSurfLast;
            int laserCloudSurfStackNum = laserCloudSurfStack->points.size();


            laserCloudCornerFromMap->clear();
            laserCloudSurfFromMap->clear();

            cornerFromMapCov.clear();
            surfFromMapCov.clear();

            // initializing
            if (!initSequence)
            {
                t_w_curr(0) = gnssMsg->eastPos;
                t_w_curr(1) = gnssMsg->northPos;
                t_w_curr(2) = gnssMsg->upPos;
                t_wmap_wodom = t_w_curr;
                tf::Quaternion q_GNSS;
                q_GNSS.setRPY(deg2rad(gnssMsg->roll), deg2rad(-gnssMsg->pitch), deg2rad(gnssMsg->azimuth));
                q_w_curr.w() = q_GNSS.w();
                q_w_curr.x() = q_GNSS.x();
                q_w_curr.y() = q_GNSS.y();
                q_w_curr.z() = q_GNSS.z();
                q_wmap_wodom = q_w_curr;
                tf::Quaternion q2;
                q2.setW(q_w_curr.w());
                q2.setX(q_w_curr.x());
                q2.setY(q_w_curr.y());
                q2.setZ(q_w_curr.z());
                tf::Matrix3x3 m(q2);
                // get angles
                double roll, pitch, yaw;
                m.getRPY(roll, pitch, yaw);
                state(0) = t_w_curr(0); state(1) = t_w_curr(1); state(2) = t_w_curr(2);
                state(3) = roll;    state(4) = pitch;   state(5) = yaw;
                initSequence = true;
                ROS_INFO("Mapping initialization finished \n");
            }
            else
            {
                state(0) += gnssMsg->east_vel*gnssMsg->dt; state(1) += gnssMsg->north_vel*gnssMsg->dt; state(2) += gnssMsg->up_vel*gnssMsg->dt;
                state(3) += deg2rad(gnssMsg->roll_inc);    state(4) += deg2rad(gnssMsg->pitch_inc);   state(5) += deg2rad(gnssMsg->azi_inc);
                Q.diagonal()<<pow(gnssMsg->eastVel_std,2),pow(gnssMsg->northVel_std,2),pow(5*gnssMsg->upVel_std,2),
                              pow(deg2rad(3*gnssMsg->roll_std),2),pow(deg2rad(3*gnssMsg->pitch_std),2),pow(deg2rad(3*gnssMsg->azi_std),2);
                P += Q;
                t_w_curr(0) = state(0); t_w_curr(1) = state(1); t_w_curr(2) = state(2);
                geometry_msgs::Quaternion qState = tf::createQuaternionMsgFromRollPitchYaw(state(3), state(4), state(5));
                q_w_curr.w() = qState.w;    q_w_curr.x() = qState.x;    q_w_curr.y() = qState.y;    q_w_curr.z() = qState.z;

                for (size_t ss = 0; ss < laserCloudCornerPDMap->size(); ss++)
                {
                    Eigen::Vector3d diffVec;
                    diffVec(0) = laserCloudCornerPDMap->points[ss].x - t_w_curr.x();
                    diffVec(1) = laserCloudCornerPDMap->points[ss].y - t_w_curr.y();
                    diffVec(2) = laserCloudCornerPDMap->points[ss].z - t_w_curr.z();

                    double diffDist = sqrt(pow(diffVec(0),2)+pow(diffVec(1),2)+pow(diffVec(0),2));
                    if (diffDist < mapMatchingRange)
                    {
                        laserCloudCornerFromMap->points.push_back(laserCloudCornerPDMap->points[ss]);
                        cornerFromMapCov.push_back(cornerMapCov[ss]);
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
                        if (diffDist < mapMatchingRange)
                        {
                            laserCloudSurfFromMap->points.push_back(laserCloudSurfPDMap->points[ss]);
                            surfFromMapCov.push_back(surfMapCov[ss]);
                        }
                    }
                }

                int laserCloudCornerFromMapNum = laserCloudCornerFromMap->points.size();
                int laserCloudSurfFromMapNum = laserCloudSurfFromMap->points.size();

                TicToc t_opt;
                TicToc t_tree;
                kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMap);
                kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMap);

                for (int iterCount = 0; iterCount < 2; iterCount++)
                {
                    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                    ceres::LocalParameterization *q_parameterization =
                        new ceres::EigenQuaternionParameterization();
                    ceres::Problem::Options problem_options;

                    ceres::Problem problem(problem_options);
                    problem.AddParameterBlock(parameters, 3);
                    problem.AddParameterBlock(parameters + 3, 4, q_parameterization);

                    TicToc t_data;
                    int corner_num = 0;
                    if (laserCloudCornerFromMapNum > 0)
                    {
                        for (int i = 0; i < laserCloudCornerStackNum; i++)
                        {
                            pointOri = laserCloudCornerStack->points[i];

                            pointAssociateToMap(&pointOri, &pointSel);
                            kdtreeCornerFromMap->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
                            double minPointSqDis = matchingThres;
                            int minPointInd = -1;
                            for (size_t s = 0; s < pointSearchInd.size(); s++)
                            {
                                if (pointSearchSqDis[s] < 6)
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

                                Eigen::Vector3d point_on_line;
                                point_on_line(0) = laserCloudCornerFromMap->points[minPointInd].x;
                                point_on_line(1) = laserCloudCornerFromMap->points[minPointInd].y;
                                point_on_line(2) = laserCloudCornerFromMap->points[minPointInd].z;
                                Eigen::Vector3d point_a, point_b;
                                point_a = unit_direction + point_on_line;
                                point_b = -unit_direction + point_on_line;
                                ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, point_a, point_b, 1.0);
                                problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 3);
                                corner_num++;
                            }
                        }
                    }
                    else
                    {
                        ROS_WARN("Map corner points are not enough");
                    }

                    int surf_num = 0;
                    if (laserCloudSurfFromMapNum > 50)
                    {
                        for (int i = 0; i < laserCloudSurfStackNum; i++)
                        {
                            pointOri = laserCloudSurfStack->points[i];
                            //double sqrtDis = pointOri.x * pointOri.x + pointOri.y * pointOri.y + pointOri.z * pointOri.z;
                            pointAssociateToMap(&pointOri, &pointSel);
                            kdtreeSurfFromMap->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);

                            double minPointSqDis = matchingThres;
                            int minPointInd = -1;
                            for (size_t s = 0; s < pointSearchInd.size(); s++)
                            {
                                if (pointSearchSqDis[s] < 6)
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
                                point_a = direction1 + point_on_surf;
                                point_b = -direction1 + point_on_surf;
                                point_c = direction2 + point_on_surf;

                                ceres::CostFunction *cost_function = LidarPlaneFactor::Create(curr_point, point_a, point_b, point_c, 1.0);
                                problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 3);
                                surf_num++;
                            }
                        }
                    }
                    else
                    {
                        ROS_WARN("Map surf points are not enough");
                    }

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
                        final_cost = summary.final_cost/(corner_num + surf_num);
                    }
                }
            }
            tf::Quaternion q2;
            q2.setW(q_w_curr.w());
            q2.setX(q_w_curr.x());
            q2.setY(q_w_curr.y());
            q2.setZ(q_w_curr.z());
            tf::Matrix3x3 m(q2);
            // get angles
            double roll, pitch, yaw;
            m.getRPY(roll, pitch, yaw);
            if (final_cost < 0.04)
            {
                R.diagonal()<<pow(0.2,2),pow(0.2,2),pow(0.2,2),pow(deg2rad(0.2),2),pow(deg2rad(0.2),2),pow(deg2rad(0.2),2),
                        pow(deg2rad(3*gnssMsg->roll_std),2),pow(deg2rad(3*gnssMsg->pitch_std),2),pow(deg2rad(3*gnssMsg->azi_std),2);
                Eigen::MatrixXd HPHTR(9,9);
                HPHTR = H*P*H.transpose() + R;
                K = P*H.transpose()*HPHTR.inverse();
                Eigen::VectorXd z(9);
                z(0) = t_w_curr(0); z(1) = t_w_curr(1); z(2) = t_w_curr(2);
                z(3) = roll;    z(4) = pitch;   z(5) = yaw;
                z(6) = deg2rad(gnssMsg->roll);  z(7) = deg2rad(-gnssMsg->pitch); z(8) = deg2rad(gnssMsg->azimuth);
                Eigen::VectorXd residual(9);
                residual = z-H*state;
                residual(5) = pi2piRad(residual(5));
                residual(8) = pi2piRad(residual(8));
                state += K*residual;
                Eigen::MatrixXd I(6,6);
                I.setIdentity();
                P = (I - K*H)*P;
            }
            else
            {
                ROS_INFO("LOAM result is bad.");
            }

#if DEBUG_MODE_MAPPING == 1
            for (size_t c = 0; c < cornerMapCov.size(); c++)
            {
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cornerMapCov[c]);
                Eigen::Vector3d mean;
                mean(0) = laserCloudCornerPDMap->points[c].x;
                mean(1) = laserCloudCornerPDMap->points[c].y;
                mean(2) = laserCloudCornerPDMap->points[c].z;
                drawEigenVector(ros::Time().fromSec(timeLaserOdometry), 1, marker_id, mean, saes.eigenvectors(), saes.eigenvalues(), eig_marker);
                marker_id = eig_marker.markers.back().id+1;
            }

            for (size_t s = 0; s < surfMapCov.size(); s++)
            {
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(surfMapCov[s]);
                Eigen::Vector3d mean;
                mean(0) = laserCloudSurfPDMap->points[s].x;
                mean(1) = laserCloudSurfPDMap->points[s].y;
                mean(2) = laserCloudSurfPDMap->points[s].z;
                drawEigenVector(ros::Time().fromSec(timeLaserOdometry), 2, marker_id, mean, saes.eigenvectors(), saes.eigenvalues(), eig_marker);
                marker_id = eig_marker.markers.back().id+1;
            }
#endif

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

            double wholeProcessTime = t_whole.toc();
            ProcessTimeMean = ProcessTimeMean*FrameNum + wholeProcessTime;
            FrameNum++;
            ProcessTimeMean /= FrameNum;

            std_msgs::Float32 float_time;
            float_time.data = (float)ProcessTimeMean;
            pubProcessTime.publish(float_time);

            std_msgs::Float32 float_cost;
            float_cost.data = (float)final_cost;
            pubCost.publish(float_cost);
            final_cost = 100;

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


            geometry_msgs::PoseStamped laserAfterMappedPose;
            laserAfterMappedPose.header = odomAftMapped.header;
            laserAfterMappedPose.pose = odomAftMapped.pose.pose;

            laserAfterMappedPath.header.stamp = odomAftMapped.header.stamp;
            laserAfterMappedPath.header.frame_id = "/body";
            laserAfterMappedPath.poses.push_back(laserAfterMappedPose);
            pubLaserAfterMappedPath.publish(laserAfterMappedPath);

            tf::Transform transform_GNSS;
            transform_GNSS.setOrigin(tf::Vector3(gnssMsg->eastPos, gnssMsg->northPos, gnssMsg->upPos));
            tf::Quaternion q_GNSS;
            q_GNSS.setRPY(deg2rad(gnssMsg->roll), deg2rad(-gnssMsg->pitch), deg2rad(gnssMsg->azimuth));
            geometry_msgs::PoseStamped gnssPose;
            gnssPose.header.frame_id = "/body";
            gnssPose.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            gnssPose.pose.position.x = gnssMsg->eastPos;
            gnssPose.pose.position.y = gnssMsg->northPos;
            gnssPose.pose.position.z = gnssMsg->upPos;
            gnssPose.pose.orientation.w = q_GNSS.w();
            gnssPose.pose.orientation.x = q_GNSS.x();
            gnssPose.pose.orientation.y = q_GNSS.y();
            gnssPose.pose.orientation.z = q_GNSS.z();
            gnssEnuPath.header.stamp = gnssPose.header.stamp;
            gnssEnuPath.header.frame_id = "/body";
            gnssEnuPath.poses.push_back(gnssPose);
            pubGnssPath.publish(gnssEnuPath);

            tf::Transform transform_KF;
            transform_KF.setOrigin(tf::Vector3(state(0), state(1), state(2)));
            tf::Quaternion q_KF;
            q_KF.setRPY(state(3), state(4), state(5));
            geometry_msgs::PoseStamped KFPose;
            KFPose.header.frame_id = "/body";
            KFPose.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            KFPose.pose.position.x = state(0);
            KFPose.pose.position.y = state(1);
            KFPose.pose.position.z = state(2);
            KFPose.pose.orientation.w = q_KF.w();
            KFPose.pose.orientation.x = q_KF.x();
            KFPose.pose.orientation.y = q_KF.y();
            KFPose.pose.orientation.z = q_KF.z();
            KalmanPath.header.stamp = KFPose.header.stamp;
            KalmanPath.header.frame_id = "/body";
            KalmanPath.poses.push_back(KFPose);
            pubKalmanPath.publish(KalmanPath);

            nav_msgs::Odometry odomKF;
            odomKF.header.frame_id = "/body";
            odomKF.child_frame_id = "/aft_KF";
            odomKF.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            odomKF.pose.pose.orientation.x = q_KF.x();
            odomKF.pose.pose.orientation.y = q_KF.y();
            odomKF.pose.pose.orientation.z = q_KF.z();
            odomKF.pose.pose.orientation.w = q_KF.w();
            odomKF.pose.pose.position.x = state(0);
            odomKF.pose.pose.position.y = state(1);
            odomKF.pose.pose.position.z = state(2);

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
            frameMsg.pose = odomKF;
            frameMsg.GNSS = *gnssMsg;
            frameMsg.frame_idx = frameCount;
            pubFrame.publish(frameMsg);


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

            transform_GNSS.setRotation(q_GNSS);
            br.sendTransform(tf::StampedTransform(transform_GNSS, ros::Time().fromSec(timeLaserOdometry), "/body", "/GNSS_INS"));
            transform_KF.setRotation(q_KF);
            br.sendTransform(tf::StampedTransform(transform_KF, ros::Time().fromSec(timeLaserOdometry), "/body", "/KF"));
#if DEBUG_MODE_MAPPING == 1
            pubEigenMarker.publish(eig_marker);
            eig_marker.markers.clear();
            marker_id = 1;
#endif

            transformUpdate();

            t_w_curr(0) = state(0); t_w_curr(1) = state(1); t_w_curr(2) = state(2);
            geometry_msgs::Quaternion qState = tf::createQuaternionMsgFromRollPitchYaw(state(3), state(4), state(5));
            q_w_curr.w() = qState.w;    q_w_curr.x() = qState.x;    q_w_curr.y() = qState.y;    q_w_curr.z() = qState.z;

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
            laserCloudSurfLastPD->clear();
            surfCov.clear();

            getCornerProbabilityDistributions(cornerTemp);

            getSurfProbabilityDistributions(surfTemp);

            updateMap();
            mMapping.unlock();
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

    P.diagonal()<<100,100,100,100,100,100;
    H(0,0) = 1; H(1,1) = 1; H(2,2) = 1;
    H(3,3) = 1; H(4,4) = 1; H(5,5) = 1;
    H(6,3) = 1; H(7,4) = 1; H(8,5) = 1;
    state.setZero();
    nh.param<float>("map_matching_range", mapMatchingRange, 80);
    nh.param<double>("matching_threshold",matchingThres, 4);

    ros::Subscriber subLaserCloudCornerLast = nh.subscribe<sensor_msgs::PointCloud2>("/odometry/laser_cloud_corner_last", 100, laserCloudCornerLastHandler);

    ros::Subscriber subLaserCloudSurfLast = nh.subscribe<sensor_msgs::PointCloud2>("/odometry/laser_cloud_surf_last", 100, laserCloudSurfLastHandler);

    ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/odometry/laser_odom_to_init", 100, laserOdometryHandler);

    ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100, laserCloudFullResHandler);

    ros::Subscriber subLaserCloudFeature = nh.subscribe<sensor_msgs::PointCloud2>("/feature/laser_feature_cluster_points", 100, laserCloudFeatureHandler);

    ros::Subscriber subGNSS = nh.subscribe<pd_loam::gnss>("/GNSS", 100, gnssHandler);

    pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/mapping/velodyne_cloud_registered", 100);

    pubLaserCloudFullResLocal = nh.advertise<sensor_msgs::PointCloud2>("/mapping/velodyne_cloud_registered_local", 100);

    pubLaserCloudFeature = nh.advertise<sensor_msgs::PointCloud2>("/mapping/aft_mapped_laser_feature_cluster_points", 100);

    pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/mapping/aft_mapped_to_init", 100);

    pubLaserAfterMappedPath = nh.advertise<nav_msgs::Path>("/mapping/aft_mapped_path", 100);

    pubGnssPath = nh.advertise<nav_msgs::Path>("/mapping/GNSS_INS_path", 100);

    pubKalmanPath = nh.advertise<nav_msgs::Path>("/mapping/KF_path", 100);

    pubEigenMarker = nh.advertise<visualization_msgs::MarkerArray>("/map_eigen", 100);

    pubFrame = nh.advertise<pd_loam::frame>("/mapping/frame", 100);

    pubProcessTime = nh.advertise<std_msgs::Float32>("/mapping_process_time", 100);

    pubCost = nh.advertise<std_msgs::Float32>("/cost", 100);
   
    std::thread mapping_process{process};

    ros::spin();

    return 0;
}
