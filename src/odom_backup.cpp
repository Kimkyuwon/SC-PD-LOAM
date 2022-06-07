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

#include <cmath>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float32.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>
#include <mutex>
#include <queue>

#include "common.h"
#include "tic_toc.h"
#include "lidarFactor.hpp"
#include "utility.h"
#include "parameter.h"
#include "integration_base.h"
#include "imufactor.h"

#define DISTORTION 1


int corner_correspondence = 0, plane_correspondence = 0;

constexpr double SCAN_PERIOD = 0.1;
constexpr double DISTANCE_SQ_THRESHOLD = 5;

int skipFrameNum = 5;
int FrameNum = 0;
bool systemInited = false;

double timeCornerPointsSharp = 0;
double timeSurfPointsFlat = 0;
double timeLaserCloudFullRes = 0;
double timeLaserCloudCluster = 0;

double ProcessTimeMean = 0;
pcl::KdTreeFLANN<pcl::PointXYZINormal>::Ptr kdtreeCornerLast(new pcl::KdTreeFLANN<pcl::PointXYZINormal>());
pcl::KdTreeFLANN<pcl::PointXYZINormal>::Ptr kdtreeSurfLast(new pcl::KdTreeFLANN<pcl::PointXYZINormal>());

pcl::PointCloud<PointType>::Ptr ClusterPoints(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr cornerPointsSharp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr cornerPointsCluster(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsFlat(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsCluster(new pcl::PointCloud<PointType>());

pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());

pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());

int laserCloudCornerLastNum = 0;
int laserCloudSurfLastNum = 0;

// Transformation from current frame to world frame
Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
Eigen::Vector3d t_w_curr(0, 0, 0);
Eigen::Quaterniond q_imu_curr(1, 0, 0, 0);
Eigen::Vector3d t_imu_curr(0, 0, 0);
Eigen::Vector3d v_imu_curr(0, 0, 0);

// q_curr_last(x, y, z, w), t_curr_last
double parameters[7] = {0, 0, 0, 1, 0, 0, 0};
double paraspeedbias[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
double pre_paraspeedbias[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

Eigen::Map<Eigen::Quaterniond> q_last_curr(parameters);
Eigen::Map<Eigen::Vector3d> t_last_curr(parameters+4);
Eigen::Map<Eigen::Vector3d> v_last_curr(paraspeedbias);
Eigen::Map<Eigen::Vector3d> ba_last_curr(paraspeedbias+3);
Eigen::Map<Eigen::Vector3d> bg_last_curr(paraspeedbias+6);

Eigen::Matrix3d R_IMU;
bool imuFirst = false;
bool imuStart = false;
double imuTimelast = 0;
Eigen::Vector3d g(0, 0, 9.805);
Eigen::Vector3d acc_0(0, 0, 0);
Eigen::Vector3d gyr_0(0, 0, 0);
Eigen::Vector3d Ba(0, 0, 0);
Eigen::Vector3d Bg(0, 0, 0);
Eigen::Vector3d sum_acc(0, 0, 0);
IntegrationBase *pre_integration;
int cnt = 0;

std::queue<sensor_msgs::PointCloud2ConstPtr> cornerSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullPointsBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> ClusterPointsBuf;
std::queue<sensor_msgs::ImuConstPtr> ImuBuf;
std::mutex mBuf;

// undistort lidar point
void TransformToStart(PointType const *const pi, PointType *const po)
{
    //interpolation ratio
    double s;
    if (DISTORTION)
        s = (pi->_PointXYZINormal::curvature - int(pi->_PointXYZINormal::curvature)) / SCAN_PERIOD;
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

// transform all lidar points to the start of the next frame

void TransformToEnd(PointType const *const pi, PointType *const po)
{
    // undistort point first
    pcl::PointXYZINormal un_point_tmp;
    TransformToStart(pi, &un_point_tmp);

    Eigen::Vector3d un_point(un_point_tmp.x, un_point_tmp.y, un_point_tmp.z);
    Eigen::Vector3d point_end = q_last_curr.inverse() * (un_point - t_last_curr);

    po->x = point_end.x();
    po->y = point_end.y();
    po->z = point_end.z();

    //Remove distortion time info
    po->_PointXYZINormal::curvature = int(pi->_PointXYZINormal::curvature);
    po->_PointXYZINormal::normal_y = pi->_PointXYZINormal::normal_y;
    po->_PointXYZINormal::normal_z = pi->_PointXYZINormal::normal_z;
}

void laserCloudSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsSharp2)
{
    mBuf.lock();
    cornerSharpBuf.push(cornerPointsSharp2);
    mBuf.unlock();
}

void laserCloudFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsFlat2)
{
    mBuf.lock();
    surfFlatBuf.push(surfPointsFlat2);
    mBuf.unlock();
}

//receive all point cloud
void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
{
    mBuf.lock();
    fullPointsBuf.push(laserCloudFullRes2);
    mBuf.unlock();
}

void laserCloudClusterHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudCluster)
{
    mBuf.lock();
    ClusterPointsBuf.push(laserCloudCluster);
    mBuf.unlock();
}

void ImuHandler(const sensor_msgs::ImuConstPtr &Imu)
{
    mBuf.lock();
    ImuBuf.push(Imu);
    mBuf.unlock();
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "laserOdometry");
    ros::NodeHandle nh;

    nh.param<int>("mapping_skip_frame", skipFrameNum, 2);

    //printf("Mapping %d Hz \n", 10 / skipFrameNum);

    ros::Subscriber subCornerPointsSharp = nh.subscribe<sensor_msgs::PointCloud2>("/feature/laser_cloud_sharp", 100, laserCloudSharpHandler);

    ros::Subscriber subSurfPointsFlat = nh.subscribe<sensor_msgs::PointCloud2>("/feature/laser_cloud_flat", 100, laserCloudFlatHandler);

    ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100, laserCloudFullResHandler);

    ros::Subscriber subClusterPoints = nh.subscribe<sensor_msgs::PointCloud2>("/feature/laser_feature_cluster_points", 100, laserCloudClusterHandler);

    ros::Subscriber subIMU = nh.subscribe<sensor_msgs::Imu>("/camera/imu", 100, ImuHandler);

    ros::Publisher pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>("/odometry/laser_cloud_corner_last", 100);

    ros::Publisher pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>("/odometry/laser_cloud_surf_last", 100);

    ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100);

    ros::Publisher pubLaserOdometry = nh.advertise<nav_msgs::Odometry>("/odometry/laser_odom_to_init", 100);

    ros::Publisher pubLaserPath = nh.advertise<nav_msgs::Path>("/odometry/laser_odom_path", 100);

    ros::Publisher pubImuPath = nh.advertise<nav_msgs::Path>("/imu_odom_path", 100);

    ros::Publisher pubProcessTime = nh.advertise<std_msgs::Float32>("/odom_process_time", 100);

    nav_msgs::Path laserPath, imuPath;

    Eigen::AngleAxisd rotX_imu(-M_PI/2, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd rotY_imu(0, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd rotZ_imu(-M_PI/2, Eigen::Vector3d::UnitZ());
    R_IMU = (rotZ_imu * rotY_imu * rotX_imu).matrix();

    int frameCount = 0;
    ros::Rate rate(100);

    while (ros::ok())
    {
        ros::spinOnce();

        if (!cornerSharpBuf.empty() && !surfFlatBuf.empty() &&
            !fullPointsBuf.empty() && !ClusterPointsBuf.empty())
        {
            timeCornerPointsSharp = cornerSharpBuf.front()->header.stamp.toSec();
            timeSurfPointsFlat = surfFlatBuf.front()->header.stamp.toSec();
            timeLaserCloudFullRes = fullPointsBuf.front()->header.stamp.toSec();
            timeLaserCloudCluster = ClusterPointsBuf.front()->header.stamp.toSec();

            if (timeCornerPointsSharp != timeLaserCloudFullRes ||
                timeSurfPointsFlat != timeLaserCloudFullRes ||
                timeLaserCloudCluster != timeLaserCloudFullRes)
            {
                //printf("unsync messeage!");
                ROS_BREAK();
            }

            mBuf.lock();

            q_imu_curr = q_w_curr;
            t_imu_curr = t_w_curr;
            pre_integration = new IntegrationBase{acc_0, gyr_0, Ba, Bg};
            while(!ImuBuf.empty())
            {
                sensor_msgs::Imu ImuMsg;
                ImuMsg = *ImuBuf.front();
                double t = ImuMsg.header.stamp.toSec();
                if (imuFirst == false)
                {
                    imuTimelast = ImuMsg.header.stamp.toSec();
                    imuFirst = true;
                }
                double dt = t - imuTimelast;
                imuTimelast = t;
                double dx = ImuMsg.linear_acceleration.x;
                double dy = ImuMsg.linear_acceleration.y;
                double dz = ImuMsg.linear_acceleration.z;
                Eigen::Vector3d linear_acceleration{dx, dy, dz};

                double rx = ImuMsg.angular_velocity.x;
                double ry = ImuMsg.angular_velocity.y;
                double rz = ImuMsg.angular_velocity.z;
                Eigen::Vector3d angular_velocity{rx, ry, rz};

                linear_acceleration = R_IMU * linear_acceleration;
                angular_velocity = R_IMU * angular_velocity;

                if (imuStart == false && cnt < 20)
                {
                    sum_acc += linear_acceleration;
                    acc_0 = linear_acceleration;
                    gyr_0 = angular_velocity;
                }
                else if(imuStart == false && cnt == 20)
                {
                    sum_acc /= 20;
                    g = sum_acc;
                    pre_integration->repropagate(Vector3d::Zero(), Vector3d::Zero());
                    imuStart = true;
                }

                if (imuStart == false)
                {
                    cnt++;
                    ImuBuf.pop();
                    continue;
                }

                pre_integration->push_back(dt, acc_0, gyr_0);
                Eigen::Vector3d un_acc_0 = q_w_curr * (acc_0 - Ba) - g;

                Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bg;
                q_imu_curr = q_w_curr * Utility::deltaQ(un_gyr * dt);

                Eigen::Vector3d un_acc_1 = q_w_curr * (linear_acceleration - Ba) - g;

                Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

                t_imu_curr = t_w_curr + dt * v_imu_curr + 0.5 * dt * dt * un_acc;
                v_imu_curr = v_imu_curr + dt * un_acc;

                acc_0 = linear_acceleration;
                gyr_0 = angular_velocity;

                cnt++;
                ImuBuf.pop();
            }
            Eigen::Matrix3d imuR = q_imu_curr.toRotationMatrix();
            Eigen::Matrix4d imuTF(Eigen::Matrix4d::Identity());
            imuTF.block(0,0,3,3) = imuR;
            imuTF(0,3) = t_imu_curr(0);    imuTF(1,3) = t_imu_curr(1);    imuTF(2,3) = t_imu_curr(2);
            Eigen::Matrix3d currR = q_w_curr.toRotationMatrix();
            Eigen::Matrix4d currTF(Eigen::Matrix4d::Identity());
            currTF.block(0,0,3,3) = currR;
            currTF(0,3) = t_w_curr(0);    currTF(1,3) = t_w_curr(1);    currTF(2,3) = t_w_curr(2);
            Eigen::Matrix4d imuDeltaTF = currTF.inverse() * imuTF;

            cornerPointsSharp->clear();
            pcl::fromROSMsg(*cornerSharpBuf.front(), *cornerPointsSharp);
            cornerSharpBuf.pop();

            surfPointsFlat->clear();
            pcl::fromROSMsg(*surfFlatBuf.front(), *surfPointsFlat);
            surfFlatBuf.pop();

            ClusterPoints->clear();
            pcl::fromROSMsg(*ClusterPointsBuf.front(), *ClusterPoints);
            cornerPointsCluster->clear();
            surfPointsCluster->clear();

            for (size_t k = 0; k < ClusterPoints->points.size(); k++)
            {
                if (ClusterPoints->points[k]._PointXYZINormal::normal_z == 1)
                {
                    cornerPointsCluster->push_back(ClusterPoints->points[k]);
                }
                else if (ClusterPoints->points[k]._PointXYZINormal::normal_z == 2)
                {
                    surfPointsCluster->push_back(ClusterPoints->points[k]);
                }
            }

            ClusterPointsBuf.pop();

            laserCloudFullRes->clear();
            pcl::fromROSMsg(*fullPointsBuf.front(), *laserCloudFullRes);
            fullPointsBuf.pop();
            mBuf.unlock();

            TicToc t_whole;
            // initializing
            if (!systemInited)
            {
                systemInited = true;
                std::cout << "Initialization finished \n";
            }
            else
            {
                int cornerPointsSharpNum = cornerPointsSharp->points.size();
                int surfPointsFlatNum = surfPointsFlat->points.size();

                TicToc t_opt;

                for (size_t opti_counter = 0; opti_counter < 1; ++opti_counter)
                {
                    corner_correspondence = 0;
                    plane_correspondence = 0;

                    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                    ceres::LocalParameterization *q_parameterization =
                        new ceres::EigenQuaternionParameterization();
                    ceres::Problem::Options problem_options;

                    ceres::Problem problem(problem_options);


                    problem.AddParameterBlock(pre_paraspeedbias, 9);
                    problem.AddParameterBlock(parameters, 4, q_parameterization);
                    problem.AddParameterBlock(parameters + 4, 3);
                    problem.AddParameterBlock(paraspeedbias, 9);

                    pcl::PointXYZINormal pointSel;
                    std::vector<int> pointSearchInd;
                    std::vector<float> pointSearchSqDis;

                    std::vector<int> pointSearchInd2;
                    std::vector<float> pointSearchSqDis2;

                    TicToc t_data;
                    Eigen::Vector3d delta_t(imuDeltaTF(0,3),imuDeltaTF(1,3),imuDeltaTF(2,3));
                    Eigen::Matrix3d delta_R = imuDeltaTF.block(0,0,3,3);
                    Eigen::Quaterniond delta_q(delta_R);
//                    parameters[0] = delta_q.x();
//                    parameters[1] = delta_q.y();
//                    parameters[2] = delta_q.z();
//                    parameters[3] = delta_q.w();
//                    parameters[4] = delta_t(0);
//                    parameters[5] = delta_t(1);
//                    parameters[6] = delta_t(2);

                    paraspeedbias[0] = v_imu_curr(0);
                    paraspeedbias[1] = v_imu_curr(1);
                    paraspeedbias[2] = v_imu_curr(2);

                    paraspeedbias[3] = Ba(0);
                    paraspeedbias[4] = Ba(1);
                    paraspeedbias[5] = Ba(2);

                    paraspeedbias[6] = Bg(0);
                    paraspeedbias[7] = Bg(1);
                    paraspeedbias[8] = Bg(2);

                    if (imuStart == true)
                    {
                        IMUFactor* imu_factor = new IMUFactor(pre_integration);
                        //problem.AddResidualBlock(imu_factor, NULL, pre_paraspeedbias, parameters, parameters + 4, paraspeedbias);
                    }
                    // find correspondence for corner features
                    for (int i = 0; i < cornerPointsSharpNum; ++i)
                    {
                        TransformToStart(&(cornerPointsSharp->points[i]), &pointSel);

                        kdtreeCornerLast->nearestKSearch(pointSel, 5, pointSearchInd2, pointSearchSqDis2);
                        double minPointSqDis2 = 3;
                        int minPointInd2 = -1;
                        for (size_t s = 0; s < pointSearchInd2.size(); s++)
                        {
                            if (pointSearchSqDis2[s] < DISTANCE_SQ_THRESHOLD)
                            {
                                double pointSqDis = (laserCloudCornerLast->points[pointSearchInd2[s]].x - pointSel.x) *
                                                    (laserCloudCornerLast->points[pointSearchInd2[s]].x - pointSel.x) +
                                                    (laserCloudCornerLast->points[pointSearchInd2[s]].y - pointSel.y) *
                                                    (laserCloudCornerLast->points[pointSearchInd2[s]].y - pointSel.y) +
                                                    (laserCloudCornerLast->points[pointSearchInd2[s]].z - pointSel.z) *
                                                    (laserCloudCornerLast->points[pointSearchInd2[s]].z - pointSel.z);

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
                            for (size_t j = 0; j < laserCloudCornerLast->points.size(); j++)
                            {
                                if (laserCloudCornerLast->points[minPointInd2]._PointXYZINormal::normal_y != laserCloudCornerLast->points[j]._PointXYZINormal::normal_y)
                                {
                                    continue;
                                }
                                clusters.push_back(laserCloudCornerLast->points[j]);
                            }
                            if (clusters.points.size() >= 5)
                            {
                                Eigen::Matrix3d covariance = getCovariance(clusters);
                                Eigen::Vector3d mean = getMean(clusters);
                                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covariance);

                                // if is indeed line feature
                                // note Eigen library sort eigenvalues in increasing order
                                Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);

                                Eigen::Vector3d curr_point(cornerPointsSharp->points[i].x,
                                                           cornerPointsSharp->points[i].y,
                                                           cornerPointsSharp->points[i].z);
                                if (saes.eigenvalues()[2] > 16 * saes.eigenvalues()[1] && saes.eigenvalues()[1] < pow(0.4,2))
                                {
                                    Eigen::Vector3d point_on_line;
                                    point_on_line(0) = mean(0);
                                    point_on_line(1) = mean(1);
                                    point_on_line(2) = mean(2);
                                    Eigen::Vector3d point_a, point_b;
                                    point_a = 0.1 * unit_direction + point_on_line;
                                    point_b = -0.1 * unit_direction + point_on_line;
                                    double s;
                                    if (DISTORTION)
                                        s = (cornerPointsSharp->points[i]._PointXYZINormal::curvature - int(cornerPointsSharp->points[i]._PointXYZINormal::curvature)) / SCAN_PERIOD;
                                    else
                                        s = 1.0;
                                    ceres::CostFunction *cost_function = LidarEdgeOdomFactor::Create(curr_point, point_a, point_b, s);
                                    problem.AddResidualBlock(cost_function, loss_function, pre_paraspeedbias, parameters, parameters + 4, paraspeedbias);
                                    corner_correspondence++;
                                }
                            }
                        }
                    }

                    // find correspondence for plane features
                    for (int i = 0; i < surfPointsFlatNum; ++i)
                    {
                        TransformToStart(&(surfPointsFlat->points[i]), &pointSel);
                        kdtreeSurfLast->nearestKSearch(pointSel, 5, pointSearchInd2, pointSearchSqDis2);

                        double minPointSqDis2 = 2;
                        int minPointInd2 = -1;
                        for (size_t s = 0; s < pointSearchInd2.size(); s++)
                        {
                            if (pointSearchSqDis2[s] < DISTANCE_SQ_THRESHOLD)
                            {
                                double pointSqDis = (laserCloudSurfLast->points[pointSearchInd2[s]].x - pointSel.x) *
                                                    (laserCloudSurfLast->points[pointSearchInd2[s]].x - pointSel.x) +
                                                    (laserCloudSurfLast->points[pointSearchInd2[s]].y - pointSel.y) *
                                                    (laserCloudSurfLast->points[pointSearchInd2[s]].y - pointSel.y) +
                                                    (laserCloudSurfLast->points[pointSearchInd2[s]].z - pointSel.z) *
                                                    (laserCloudSurfLast->points[pointSearchInd2[s]].z - pointSel.z);

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
                            for (size_t j = 0; j < laserCloudSurfLast->points.size(); j++)
                            {
                                if (laserCloudSurfLast->points[minPointInd2]._PointXYZINormal::normal_y != laserCloudSurfLast->points[j]._PointXYZINormal::normal_y)
                                {
                                    continue;
                                }
                                clusters.push_back(laserCloudSurfLast->points[j]);
                            }

                            if (clusters.points.size() >= 5)
                            {
                                Eigen::Matrix3d covariance = getCovariance(clusters);
                                Eigen::Vector3d mean = getMean(clusters);
                                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covariance);
                                if (saes.eigenvalues()[0] < pow(0.05,2))
                                {
                                    // if is indeed line feature
                                    // note Eigen library sort eigenvalues in increasing order
                                    Eigen::Vector3d direction1 = saes.eigenvectors().col(2);
                                    Eigen::Vector3d direction2 = saes.eigenvectors().col(1);

                                    Eigen::Vector3d curr_point(surfPointsFlat->points[i].x,
                                                               surfPointsFlat->points[i].y,
                                                               surfPointsFlat->points[i].z);

                                    Eigen::Vector3d point_on_surf;
                                    point_on_surf = mean;
                                    Eigen::Vector3d point_a, point_b, point_c;
                                    point_a = 0.1 * direction1 + point_on_surf;
                                    point_b = -0.1 * direction1 + point_on_surf;
                                    point_c = 0.1 * direction2 + point_on_surf;

                                    double s;
                                    if (DISTORTION)
                                        s = (surfPointsFlat->points[i]._PointXYZINormal::curvature - int(surfPointsFlat->points[i]._PointXYZINormal::curvature)) / SCAN_PERIOD;
                                    else
                                        s = 1.0;
                                    ceres::CostFunction *cost_function = LidarPlaneOdomFactor::Create(curr_point, point_a, point_b, point_c, s);
                                    problem.AddResidualBlock(cost_function, loss_function, pre_paraspeedbias, parameters, parameters + 4, paraspeedbias);
                                    plane_correspondence++;
                                }
                            }
                        }
                    }

                    //printf("coner_correspondance %d, plane_correspondence %d \n", corner_correspondence, plane_correspondence);
                    //printf("data association time %f ms \n", t_data.toc());

                    if ((corner_correspondence + plane_correspondence) < 10)
                    {
                        //printf("less correspondence! *************************************************\n");
                    }

                    TicToc t_solver;
                    ceres::Solver::Options options;
                    options.linear_solver_type = ceres::DENSE_QR;
                    options.max_num_iterations = 4;
                    options.minimizer_progress_to_stdout = false;
                    ceres::Solver::Summary summary;
                    ceres::Solve(options, &problem, &summary);
                }
                t_w_curr = t_w_curr + q_w_curr * t_last_curr;
                q_w_curr = q_w_curr * q_last_curr;

                v_imu_curr(0) = paraspeedbias[0];
                v_imu_curr(1) = paraspeedbias[1];
                v_imu_curr(2) = paraspeedbias[2];

                Ba(0) = paraspeedbias[3];
                Ba(1) = paraspeedbias[4];
                Ba(2) = paraspeedbias[5];

                Bg(0) = paraspeedbias[6];
                Bg(1) = paraspeedbias[7];
                Bg(2) = paraspeedbias[8];

                pre_paraspeedbias[0] = paraspeedbias[0];
                pre_paraspeedbias[1] = paraspeedbias[1];
                pre_paraspeedbias[2] = paraspeedbias[2];
                pre_paraspeedbias[3] = paraspeedbias[3];
                pre_paraspeedbias[4] = paraspeedbias[4];
                pre_paraspeedbias[5] = paraspeedbias[5];
                pre_paraspeedbias[6] = paraspeedbias[6];
                pre_paraspeedbias[7] = paraspeedbias[7];
                pre_paraspeedbias[8] = paraspeedbias[8];
            }
            delete pre_integration;
            TicToc t_pub;

            // publish odometry
            nav_msgs::Odometry laserOdometry;
            laserOdometry.header.frame_id = "/body";
            laserOdometry.child_frame_id = "/laser_odom";
            laserOdometry.header.stamp = ros::Time().fromSec(timeLaserCloudCluster);
            laserOdometry.pose.pose.orientation.x = q_w_curr.x();
            laserOdometry.pose.pose.orientation.y = q_w_curr.y();
            laserOdometry.pose.pose.orientation.z = q_w_curr.z();
            laserOdometry.pose.pose.orientation.w = q_w_curr.w();
            laserOdometry.pose.pose.position.x = t_w_curr.x();
            laserOdometry.pose.pose.position.y = t_w_curr.y();
            laserOdometry.pose.pose.position.z = t_w_curr.z();
            pubLaserOdometry.publish(laserOdometry);

            geometry_msgs::PoseStamped laserPose;
            laserPose.header = laserOdometry.header;
            laserPose.pose = laserOdometry.pose.pose;
            laserPath.header.stamp = laserOdometry.header.stamp;
            laserPath.poses.push_back(laserPose);
            laserPath.header.frame_id = "/body";
            pubLaserPath.publish(laserPath);

            nav_msgs::Odometry imuOdometry;
            imuOdometry.header.frame_id = "/body";
            imuOdometry.child_frame_id = "/imu_odom";
            imuOdometry.header.stamp = ros::Time().fromSec(timeLaserCloudCluster);
            imuOdometry.pose.pose.orientation.x = q_imu_curr.x();
            imuOdometry.pose.pose.orientation.y = q_imu_curr.y();
            imuOdometry.pose.pose.orientation.z = q_imu_curr.z();
            imuOdometry.pose.pose.orientation.w = q_imu_curr.w();
            imuOdometry.pose.pose.position.x = t_imu_curr.x();
            imuOdometry.pose.pose.position.y = t_imu_curr.y();
            imuOdometry.pose.pose.position.z = t_imu_curr.z();

            geometry_msgs::PoseStamped imuPose;
            imuPose.header = imuOdometry.header;
            imuPose.pose = imuOdometry.pose.pose;
            imuPath.header.stamp = imuOdometry.header.stamp;
            imuPath.poses.push_back(imuPose);
            imuPath.header.frame_id = "/body";
            pubImuPath.publish(imuPath);

            pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsCluster;
            cornerPointsCluster = laserCloudCornerLast;
            laserCloudCornerLast = laserCloudTemp;

            laserCloudTemp = surfPointsCluster;
            surfPointsCluster = laserCloudSurfLast;
            laserCloudSurfLast = laserCloudTemp;

            laserCloudCornerLastNum = laserCloudCornerLast->points.size();
            laserCloudSurfLastNum = laserCloudSurfLast->points.size();
            //std::cout << "the size of corner last is " << laserCloudCornerLastNum << ", and the size of surf last is " << laserCloudSurfLastNum << '\n';

            kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
            kdtreeSurfLast->setInputCloud(laserCloudSurfLast);

            if (frameCount % skipFrameNum == 0)
            {
                frameCount = 0;

                sensor_msgs::PointCloud2 laserCloudCornerLast2;
                pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
                laserCloudCornerLast2.header.stamp = ros::Time().fromSec(timeLaserCloudCluster);
                laserCloudCornerLast2.header.frame_id = "/body";
                pubLaserCloudCornerLast.publish(laserCloudCornerLast2);

                sensor_msgs::PointCloud2 laserCloudSurfLast2;
                pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
                laserCloudSurfLast2.header.stamp = ros::Time().fromSec(timeLaserCloudCluster);
                laserCloudSurfLast2.header.frame_id = "/body";
                pubLaserCloudSurfLast.publish(laserCloudSurfLast2);

                sensor_msgs::PointCloud2 laserCloudFullRes3;
                pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
                laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeLaserCloudCluster);
                laserCloudFullRes3.header.frame_id = "/body";
                pubLaserCloudFullRes.publish(laserCloudFullRes3);
            }
            //printf("publication time %f ms \n", t_pub.toc());
            //printf("whole laserOdometry time %f ms \n", t_whole.toc());
            if(t_whole.toc() > 100)
                ROS_WARN("odometry process over 100ms");

            frameCount++;

            double wholeProcessTime = t_whole.toc();
            ProcessTimeMean = ProcessTimeMean*FrameNum + wholeProcessTime;
            FrameNum++;
            ProcessTimeMean /= FrameNum;

            std_msgs::Float32 float_time;
            float_time.data = (float)ProcessTimeMean;
            pubProcessTime.publish(float_time);
        }
        rate.sleep();
    }
    return 0;
}
