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

#define DEBUG_MODE_FEATURE 1

#include <cmath>
#include <vector>
#include <string>
#include "common.h"
#include "tic_toc.h"
#include <nav_msgs/Odometry.h>
#include <opencv/cv.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/search.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <novatel_gps_msgs/Inspvax.h>
#include <std_msgs/Float32.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <jsk_rviz_plugins/OverlayText.h>
#include "pd_loam/gnss.h"

using std::atan2;
using std::cos;
using std::sin;

std::string LIDAR_TYPE;

const double scanPeriod = 0.1;
const int systemDelay = 0;

int systemInitCount = 0;
bool systemInited = false;
int N_SCANS = 0;
const float CURVATURE_THRESHOLD = 0.1;
float cloudCurvature[200000];
int cloudSortInd[200000];
int cloudNeighborPicked[200000];
int cloudLabel[200000];
int cloudOcclusion[200000];
float cloudAlphaAngle[200000];
int SharpEdgeNum, LessSharpEdgeNum, FlatSurfNum, LessFlatSurfNum;

Eigen::Vector3d llh, xyz_origin, enu;
Eigen::Vector3f rpy, velo, bodyVelo, pos_std, rpy_std, velo_std, rpy_prev, rpy_inc;
std::string gnssStatus, insStatus;
double gnssTime;
double clustering_size = 0;
pcl::PointCloud<pcl::PointXYZINormal> Cluster_points;

int FrameNum = 0;
double ProcessTimeMean = 0;
double GNSS_dt, GNSS_prevTime;

bool comp (int i,int j) { return (cloudCurvature[i]<cloudCurvature[j]); }

ros::Publisher pubLaserCloudIn;
ros::Publisher pubLaserCloud;
ros::Publisher pubCornerPointsSharp;
ros::Publisher pubSurfPointsFlat;
ros::Publisher pubRemovePoints;
ros::Publisher pubClusterPoints;
ros::Publisher pubGNSS;
ros::Publisher pubProcessTime;
ros::Publisher pubgnssText;
std::vector<ros::Publisher> pubEachScan;
jsk_rviz_plugins::OverlayText textMsg;

bool PUB_EACH_LINE = false;

double MINIMUM_RANGE = 0.1;

template <typename PointT>
void removeClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in,
                              pcl::PointCloud<PointT> &cloud_out, float thres)
{
    if (&cloud_in != &cloud_out)
    {
        cloud_out.header = cloud_in.header;
        cloud_out.points.resize(cloud_in.points.size());
    }

    size_t j = 0;

    for (size_t i = 0; i < cloud_in.points.size(); ++i)
    {
        if (cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y + cloud_in.points[i].z * cloud_in.points[i].z < thres * thres)
            continue;
        cloud_out.points[j] = cloud_in.points[i];
        j++;
    }
    if (j != cloud_in.points.size())
    {
        cloud_out.points.resize(j);
    }

    cloud_out.height = 1;
    cloud_out.width = static_cast<uint32_t>(j);
    cloud_out.is_dense = true;
}

void clusteringCornerPointCloud(const pcl::PointCloud<PointType> &cornerPointsLessSharp)
{    
    std::vector<pcl::PointCloud<PointType>> corner_clusterPC;
    std::vector<int> clusterPicked(cornerPointsLessSharp.size(), 0);

    pcl::PointCloud<PointType>::Ptr cluster_points (new pcl::PointCloud<PointType>());
    int cnt = 0;
    for (size_t i = 0; i < cornerPointsLessSharp.size() - 1; i++)
    {
        if (clusterPicked[i] == 1)  continue;

        cluster_points->clear();
        cluster_points->push_back(cornerPointsLessSharp.points[i]);
        clusterPicked[i] = 1;
        float dist_Threshold = clustering_size;
        for (size_t jj = i + 1; jj < cornerPointsLessSharp.points.size(); jj++)
        {
            PointType temp_curr_point = cornerPointsLessSharp.points[i];
            PointType temp_comp_point = cornerPointsLessSharp.points[jj];
            float diffX = temp_curr_point.x - temp_comp_point.x;
            float diffY = temp_curr_point.y - temp_comp_point.y;

            float diffDist = sqrt(diffX * diffX + diffY * diffY);

            if (diffDist < dist_Threshold && clusterPicked[jj] != 1)
            {
                clusterPicked[jj] = 1;
                cluster_points->push_back(cornerPointsLessSharp.points[jj]);
            }
        }
        if (cluster_points->size() >= 5)
        {
            pcl::KdTreeFLANN<pcl::PointXYZINormal>::Ptr kdtreeCluster(new pcl::KdTreeFLANN<pcl::PointXYZINormal>());
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;
            kdtreeCluster->setInputCloud(cluster_points);
            std::vector<int> tmpkdtreePicked(cluster_points->points.size(), 0);
            pcl::PointCloud<PointType> tmp_kdtree_cluster;
            for (auto t = 0; t < cluster_points->size(); t++)
            {
                if (tmpkdtreePicked[t] == 1)  continue;
                cluster_points->points[t]._PointXYZINormal::normal_y = cnt;
                cluster_points->points[t]._PointXYZINormal::normal_z = 1;
                bool cluster_flag = false;
                tmp_kdtree_cluster.clear();
                tmp_kdtree_cluster.push_back(cluster_points->points[t]);
                tmpkdtreePicked[t] = 1;
                kdtreeCluster->nearestKSearch(cluster_points->points[t], cluster_points->size(), pointSearchInd, pointSearchSqDis);
                for (auto ii = 0; ii < cluster_points->size(); ii++)
                {
                    if (tmpkdtreePicked[pointSearchInd[ii]] == 1 || pointSearchInd[ii] == t)   continue;

                    cluster_points->points[pointSearchInd[ii]]._PointXYZINormal::normal_y = cnt;
                    cluster_points->points[pointSearchInd[ii]]._PointXYZINormal::normal_z = 1;
                    tmp_kdtree_cluster.push_back(cluster_points->points[pointSearchInd[ii]]);

                    tmpkdtreePicked[pointSearchInd[ii]] = 1;
                    if (tmp_kdtree_cluster.size() < 5) continue;

                    Eigen::Matrix3d cov = getCovariance(tmp_kdtree_cluster);
                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cov);
                    Eigen::Vector3d eigVec = saes.eigenvectors().col(2);
                    double atanAngle = atan2(eigVec(2),sqrt(pow(eigVec(0),2)+pow(eigVec(1),2)));
                    if (saes.eigenvalues()[2] > 4*saes.eigenvalues()[1] && fabs(rad2deg(atanAngle)) > 60)
                    {
                        cluster_flag = true;
                        break;
                    }
                    cluster_flag = false;
                }
                if (cluster_flag == true)
                {
                    Cluster_points += tmp_kdtree_cluster;
                    cnt = cnt + 1;
                }
            }
        }
    }
}

void clusteringSurfPointCloud(const pcl::PointCloud<PointType> &surfPointsLessFlat)
{
    pcl::PointCloud<PointType>::Ptr tmp_cluster_points (new pcl::PointCloud<PointType>());
    std::vector<int> tmpclusterPicked(surfPointsLessFlat.points.size(), 0);
    int cnt = 0;
    for (size_t kk = 0; kk < surfPointsLessFlat.points.size(); kk++)
    {
        if (tmpclusterPicked[kk] == 1)  continue;

        tmp_cluster_points->clear();
        tmp_cluster_points->push_back(surfPointsLessFlat.points[kk]);
        tmpclusterPicked[kk] = 1;
        float dist_Threshold = clustering_size;
        for (size_t jj = kk + 1; jj < surfPointsLessFlat.points.size(); jj++)
        {
            PointType temp_curr_point = surfPointsLessFlat.points[kk];
            PointType temp_comp_point = surfPointsLessFlat.points[jj];
            float diffX = temp_curr_point.x - temp_comp_point.x;
            float diffY = temp_curr_point.y - temp_comp_point.y;
            float diffZ = temp_curr_point.z - temp_comp_point.z;

            float diffDist = sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ);

            if (diffDist < dist_Threshold && tmpclusterPicked[jj] != 1)
            {
                tmpclusterPicked[jj] = 1;
                tmp_cluster_points->push_back(surfPointsLessFlat.points[jj]);
            }
        }

        if (tmp_cluster_points->size() >= 5)
        {
            pcl::KdTreeFLANN<pcl::PointXYZINormal>::Ptr kdtreeCluster(new pcl::KdTreeFLANN<pcl::PointXYZINormal>());
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;
            kdtreeCluster->setInputCloud(tmp_cluster_points);
            std::vector<int> tmpkdtreePicked(tmp_cluster_points->points.size(), 0);
            pcl::PointCloud<PointType> tmp_kdtree_cluster;
            for (auto t = 0; t < tmp_cluster_points->size(); t++)
            {
                if (tmpkdtreePicked[t] == 1)  continue;
                tmp_cluster_points->points[t]._PointXYZINormal::normal_y = cnt;
                tmp_cluster_points->points[t]._PointXYZINormal::normal_z = 2;
                bool cluster_flag = false;
                tmp_kdtree_cluster.clear();
                tmp_kdtree_cluster.push_back(tmp_cluster_points->points[t]);
                tmpkdtreePicked[t] = 1;
                kdtreeCluster->nearestKSearch(tmp_cluster_points->points[t], tmp_cluster_points->size(), pointSearchInd, pointSearchSqDis);
                for (auto ii = 0; ii < tmp_cluster_points->size(); ii++)
                {
                    if (tmpkdtreePicked[pointSearchInd[ii]] == 1 || pointSearchSqDis[ii] > clustering_size || pointSearchInd[ii] == t)   continue;

                    tmp_cluster_points->points[pointSearchInd[ii]]._PointXYZINormal::normal_y = cnt;
                    tmp_cluster_points->points[pointSearchInd[ii]]._PointXYZINormal::normal_z = 2;
                    tmp_kdtree_cluster.push_back(tmp_cluster_points->points[pointSearchInd[ii]]);

                    tmpkdtreePicked[pointSearchInd[ii]] = 1;
                    if (tmp_kdtree_cluster.size() < 5) continue;

                    Eigen::Matrix3d cov = getCovariance(tmp_kdtree_cluster);
                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cov);

                    if (saes.eigenvalues()[0] > pow(0.1,2))
                    {
                        tmp_kdtree_cluster.points.pop_back();
                        tmpkdtreePicked[pointSearchInd[ii]] = 0;
                        break;
                    }
                    cluster_flag = true;
                }
                if (cluster_flag == true)
                {
                    Cluster_points += tmp_kdtree_cluster;
                    cnt = cnt + 1;
                }
            }
        }
    }
}

void gnssHandler(const novatel_gps_msgs::InspvaxConstPtr &gnssMsg)
{
    gnssTime = gnssMsg->header.stamp.toSec();
    gnssStatus = gnssMsg->position_type;
    insStatus = gnssMsg->ins_status;
    llh(0) = gnssMsg->latitude; llh(1) = gnssMsg->longitude;    llh(2) = gnssMsg->altitude;
    rpy(0) = gnssMsg->roll; rpy(1) = gnssMsg->pitch;    rpy(2) = 90-gnssMsg->azimuth;
    rpy = pi2pi(rpy);
    velo(0) = gnssMsg->east_velocity;   velo(1) = gnssMsg->north_velocity;  velo(2) = gnssMsg->up_velocity;
    pos_std(0) = gnssMsg->latitude_std; pos_std(1) = gnssMsg->longitude_std;    pos_std(2) = gnssMsg->altitude_std;
    rpy_std(0) = gnssMsg->roll_std; rpy_std(1) = gnssMsg->pitch_std;    rpy_std(2) = gnssMsg->azimuth_std;
    velo_std(0) = gnssMsg->east_velocity_std; velo_std(1) = gnssMsg->north_velocity_std;    velo_std(2) = gnssMsg->up_velocity_std;

    Eigen::Matrix3f rotation;
    rotation = Eigen::AngleAxisf(deg2rad(rpy(0)), Eigen::Vector3f::UnitX())
             * Eigen::AngleAxisf(-deg2rad(rpy(1)), Eigen::Vector3f::UnitY())
             * Eigen::AngleAxisf(deg2rad(rpy(2))/* + M_PI/2*/, Eigen::Vector3f::UnitZ());
    bodyVelo = rotation.inverse() * velo;
    if (systemInited == true)
    {
        Eigen::Vector3d xyz = llh2xyz(llh);
        enu = xyz2enu(xyz, xyz_origin);
    }
}

void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
    if (!systemInited && (gnssStatus == "INS_RTKFLOAT" || gnssStatus == "INS_RTKFIXED" || gnssStatus == "INS_PSRDIFF") && insStatus == "INS_SOLUTION_GOOD")
    {
        std::cout<<"system start."<<std::endl;
        xyz_origin = llh2xyz(llh);
        rpy_prev = rpy;
        GNSS_prevTime = gnssTime;
        systemInited = true;
    }
    else if (!systemInited)    return;

    TicToc t_whole;
    GNSS_dt = gnssTime - GNSS_prevTime;
    GNSS_prevTime = gnssTime;
    rpy_inc = pi2pi(rpy - rpy_prev);
    rpy_prev = rpy;
    std::vector<int> scanStartInd(N_SCANS, 0);
    std::vector<int> scanEndInd(N_SCANS, 0);
    std::vector<float> alphaAngle(N_SCANS, 0);

    pcl::PointCloud<pcl::PointXYZI> laserCloudIn;
    pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);
    std::vector<int> indices;

    pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);
    removeClosedPointCloud(laserCloudIn, laserCloudIn, MINIMUM_RANGE);


    int cloudSize = laserCloudIn.points.size();
    float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
    float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y,
                          laserCloudIn.points[cloudSize - 1].x) +
                   2 * M_PI;

    if (endOri - startOri > 3 * M_PI)
    {
        endOri -= 2 * M_PI;
    }
    else if (endOri - startOri < M_PI)
    {
        endOri += 2 * M_PI;
    }
    ////printf("end Ori %f\n", endOri);

    bool halfPassed = false;
    int count = cloudSize;
    PointType point;
    std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCANS);
    for (int i = 0; i < cloudSize; i++)
    {
        point.x = laserCloudIn.points[i].x;
        point.y = laserCloudIn.points[i].y;
        point.z = laserCloudIn.points[i].z;
        point.intensity = laserCloudIn.points[i].intensity;

        float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
        int scanID = 0;

        if (LIDAR_TYPE == "VLP16" && N_SCANS == 16)
        {
            scanID = int((angle + 15) / 2 + 0.5);
            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else if (LIDAR_TYPE == "HDL32" && N_SCANS == 32)
        {
            scanID = int((angle + 92.0/3.0) * 3.0 / 4.0);
            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                count--;
                continue;
            }
        }
        // HDL64 (e.g., KITTI)
        else if (LIDAR_TYPE == "HDL64" && N_SCANS == 64)
        {
            if (angle >= -8.83)
                scanID = int((2 - angle) * 3.0 + 0.5);
            else
                scanID = N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);

            if (angle < -24.33 || scanID > 40 || scanID < 0)
            {
                count--;
                continue;
            }
        }
        // Ouster OS1-64 (e.g., MulRan)model_outlier_removal
        else if (LIDAR_TYPE == "OS1-64" && N_SCANS == 64)
        {
            scanID = int((angle + 22.5) / 2 + 0.5); // ouster os1-64 vfov is [-22.5, 22.5] see https://ouster.com/products/os1-lidar-sensor/
            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else
        {
            //printf("wrong scan number\n");
            ROS_BREAK();
        }
        ////printf("angle %f scanID %d \n", angle, scanID);

        float ori = -atan2(point.y, point.x);
        if (!halfPassed)
        {
            if (ori < startOri - M_PI / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > startOri + M_PI * 3 / 2)
            {
                ori -= 2 * M_PI;
            }

            if (ori - startOri > M_PI)
            {
                halfPassed = true;
            }
        }
        else
        {
            ori += 2 * M_PI;
            if (ori < endOri - M_PI * 3 / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > endOri + M_PI / 2)
            {
                ori -= 2 * M_PI;
            }
        }

        float relTime = (ori - startOri) / (endOri - startOri);
        point._PointXYZINormal::curvature = scanID + scanPeriod * relTime;
        point._PointXYZINormal::normal_x = ori;
        double scale = (point._PointXYZINormal::curvature - int(point._PointXYZINormal::curvature)) / scanPeriod;
        if (scale > 1)
        {
            scale -= int(scale);
        }
        Eigen::Vector3f linearInc = bodyVelo * scanPeriod * scale;
        Eigen::Vector3f angInc = rpy_inc * scale;
        Eigen::Matrix3f R;
        R = Eigen::AngleAxisf(deg2rad(angInc(0)), Eigen::Vector3f::UnitX())
          * Eigen::AngleAxisf(-deg2rad(angInc(1)), Eigen::Vector3f::UnitY())
          * Eigen::AngleAxisf(-deg2rad(angInc(2)), Eigen::Vector3f::UnitZ());
        pcl::PointXYZ tempPoint;
        tempPoint.x = R(0,0) * point.x + R(0,1) * point.y + R(0,2) * point.z + linearInc(0);
        tempPoint.y = R(1,0) * point.x + R(1,1) * point.y + R(1,2) * point.z + linearInc(1);
        tempPoint.z = R(2,0) * point.x + R(2,1) * point.y + R(2,2) * point.z + linearInc(2);
        point.x = tempPoint.x;  point.y = tempPoint.y;  point.z = tempPoint.z;
        laserCloudScans[scanID].push_back(point);
    }
    
    cloudSize = count;

    //printf("points size %d \n", cloudSize);

    pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
    for (int i = 0; i < N_SCANS; i++)
    {
        scanStartInd[i] = laserCloud->size() + 5;
        *laserCloud += laserCloudScans[i];
        scanEndInd[i] = laserCloud->size() - 6;
        alphaAngle[i] = (endOri - startOri)/laserCloudScans[i].size();
    }

    for (int i = 5; i < cloudSize - 5; i++)
    {
        float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x + laserCloud->points[i + 3].x + laserCloud->points[i + 4].x + laserCloud->points[i + 5].x;
        float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y + laserCloud->points[i + 3].y + laserCloud->points[i + 4].y + laserCloud->points[i + 5].y;
        float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z + laserCloud->points[i + 3].z + laserCloud->points[i + 4].z + laserCloud->points[i + 5].z;

        float neighborDist1 = sqrt(
                pow((laserCloud->points[i - 1].x - laserCloud->points[i].x),2) +
                pow((laserCloud->points[i - 1].y - laserCloud->points[i].y),2) +
                pow((laserCloud->points[i - 1].z - laserCloud->points[i].z),2));
        float neighborDist2 = sqrt(
                pow((laserCloud->points[i + 1].x - laserCloud->points[i].x),2) +
                pow((laserCloud->points[i + 1].y - laserCloud->points[i].y),2) +
                pow((laserCloud->points[i + 1].z - laserCloud->points[i].z),2));

        float range1 = sqrt(pow(laserCloud->points[i].x,2) + pow(laserCloud->points[i].y,2) + pow(laserCloud->points[i].z,2));
        float range2 = sqrt(pow(laserCloud->points[i-1].x,2) + pow(laserCloud->points[i-1].y,2) + pow(laserCloud->points[i-1].z,2));
        float range3 = sqrt(pow(laserCloud->points[i+1].x,2) + pow(laserCloud->points[i+1].y,2) + pow(laserCloud->points[i+1].z,2));

        float d1 = fmax(range1, range2);
        float d2 = fmin(range1, range2);
        float alpha = alphaAngle[int(laserCloud->points[i]._PointXYZINormal::curvature)];
        float angle1 = atan2(d2*sin(alpha), (d1 -d2*cos(alpha)));
        d1 = fmax(range1, range3);
        d2 = fmin(range1, range3);
        float angle2 = atan2(d2*sin(alpha), (d1 -d2*cos(alpha)));

        cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;
        cloudSortInd[i] = i;
        cloudNeighborPicked[i] = 0;
        cloudLabel[i] = 0;
        if (neighborDist1 > 0.1 || neighborDist2 > 0.1)
        {
            cloudOcclusion[i] = 1;
        }
        else
        {
            cloudOcclusion[i] = 0;
        }
        cloudAlphaAngle[i] = rad2deg(fmin(angle1, angle2));
    }


    TicToc t_pts;

    pcl::PointCloud<PointType> cornerPointsSharp;
    pcl::PointCloud<PointType> cornerPointsLessSharp;
    pcl::PointCloud<PointType> surfPointsFlat;
    pcl::PointCloud<PointType> surfPointsLessFlat;

    float t_q_sort = 0;
    for (int i = 0; i < N_SCANS; i++)
    {
        if( scanEndInd[i] - scanStartInd[i] < 6)
            continue;
        pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType>::Ptr cornerPointsLessSharpScan(new pcl::PointCloud<PointType>);
        for (int j = 0; j < 6; j++)
        {
            int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6;
            int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;

            TicToc t_tmp;
            std::sort (cloudSortInd + sp, cloudSortInd + ep + 1, comp);
            t_q_sort += t_tmp.toc();

            int largestPickedNum = 0;
            for (int k = ep; k >= sp; k--)
            {
                int ind = cloudSortInd[k];                

                if (cloudNeighborPicked[ind] == 0 &&
                    cloudCurvature[ind] > CURVATURE_THRESHOLD &&
                    cloudOcclusion[ind] == 0 && cloudAlphaAngle[ind] > 40 &&
                    ind != sp && ind != ep)
                {

                    largestPickedNum++;
                    if (largestPickedNum <= SharpEdgeNum)
                    {
                        cloudLabel[ind] = 2;
                        cornerPointsSharp.push_back(laserCloud->points[ind]);
                        cornerPointsLessSharpScan->push_back(laserCloud->points[ind]);
                    }
                    else if (largestPickedNum <= LessSharpEdgeNum)
                    {
                        cloudLabel[ind] = 1;
                        cornerPointsLessSharpScan->push_back(laserCloud->points[ind]);
                    }
                    else
                    {
                        break;
                    }

                    cloudNeighborPicked[ind] = 1;

                    for (int l = 1; l <= 5; l++)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

            int smallestPickedNum = 0;
            for (int k = sp; k <= ep; k++)
            {
                int ind = cloudSortInd[k];

                if (cloudNeighborPicked[ind] == 0 &&
                    cloudCurvature[ind] < CURVATURE_THRESHOLD &&
                    cloudAlphaAngle[ind] > 40)
                {

                    cloudLabel[ind] = -1;
                    surfPointsFlat.push_back(laserCloud->points[ind]);

                    smallestPickedNum++;

                    if (smallestPickedNum >= FlatSurfNum)
                    {
                        break;
                    }

                    cloudNeighborPicked[ind] = 1;
                    for (int l = 1; l <= 5; l++)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

            int cnt = int((ep-sp)/LessFlatSurfNum) + 1;
            int k = sp;
            while(k <= ep)
            {
                if (cloudLabel[k] <= 0)
                {
                    surfPointsLessFlatScan->push_back(laserCloud->points[k]);
                }
                k += cnt;
            }
        }

        pcl::VoxelGrid<PointType> downSizeFilter;
        downSizeFilter.setInputCloud(surfPointsLessFlatScan);
        downSizeFilter.setLeafSize(0.4, 0.4, 0.4);
        downSizeFilter.filter(*surfPointsLessFlatScan);
        surfPointsLessFlat += *surfPointsLessFlatScan;

        downSizeFilter.setInputCloud(cornerPointsLessSharpScan);
        downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
        downSizeFilter.filter(*cornerPointsLessSharpScan);
        cornerPointsLessSharp += *cornerPointsLessSharpScan;
    }

#if DEBUG_MODE_FEATURE == 1
    textMsg.text = "GNSS Status : " + gnssStatus + "\n" + "INS Status : " + insStatus + "\n" +  "GNSS dt : " + std::to_string(GNSS_dt) + "\n" +
            "Absolute Position : " + std::to_string(llh(0)) + ", " + std::to_string(llh(1)) + ", " + std::to_string(llh(2)) + "\n" +
            "ENU Position : " + std::to_string(enu(0)) + ", " + std::to_string(enu(1)) + ", " + std::to_string(enu(2)) + "\n" +
            "Attitude : " + std::to_string(rpy(0)) + ", " + std::to_string(rpy(1)) + ", " + std::to_string(rpy(2)) + "\n" +
            "Attitude Incremental : " + std::to_string(rpy_inc(0)) + ", " + std::to_string(rpy_inc(1)) + ", " + std::to_string(rpy_inc(2)) + "\n" +
            "Velocity : " + std::to_string(velo(0)) + ", " + std::to_string(velo(1)) + ", " + std::to_string(velo(2)) + "\n" +
            "Body Frame Velocity : " + std::to_string(bodyVelo(0)) + ", " + std::to_string(bodyVelo(1)) + ", " + std::to_string(bodyVelo(2)) + "\n" +
            "Position STD : " + std::to_string(pos_std(0)) + ", " + std::to_string(pos_std(1)) + ", " + std::to_string(pos_std(2)) + "\n" +
            "Attitude STD : " + std::to_string(rpy_std(0)) + ", " + std::to_string(rpy_std(1)) + ", " + std::to_string(rpy_std(2)) + "\n" +
            "Velocity STD : " + std::to_string(velo_std(0)) + ", " + std::to_string(velo_std(1)) + ", " + std::to_string(velo_std(2));
    pubgnssText.publish(textMsg);
#endif

    pd_loam::gnss gnssMsg;
    gnssMsg.header.stamp = laserCloudMsg->header.stamp;
    gnssMsg.gnss_status = gnssStatus;   gnssMsg.ins_status = insStatus; gnssMsg.dt = GNSS_dt;
    gnssMsg.lat = llh(0);   gnssMsg.lon = llh(1);   gnssMsg.alt = llh(2);
    gnssMsg.eastPos = enu(0);   gnssMsg.northPos = enu(1);  gnssMsg.upPos = enu(2);
    gnssMsg.roll = rpy(0);   gnssMsg.pitch = rpy(1);   gnssMsg.azimuth = rpy(2);
    gnssMsg.roll_inc = rpy_inc(0);  gnssMsg.pitch_inc = rpy_inc(1);  gnssMsg.azi_inc = rpy_inc(2);
    gnssMsg.east_vel = velo(0);   gnssMsg.north_vel = velo(1);   gnssMsg.up_vel = velo(2);
    gnssMsg.x_vel = bodyVelo(0);    gnssMsg.y_vel = bodyVelo(1);    gnssMsg.z_vel = bodyVelo(2);
    gnssMsg.eastPos_std = pos_std(0);   gnssMsg.northPos_std = pos_std(1);   gnssMsg.upPos_std = pos_std(2);
    gnssMsg.roll_std = rpy_std(0);   gnssMsg.pitch_std = rpy_std(1);   gnssMsg.azi_std = rpy_std(2);
    gnssMsg.eastVel_std = velo_std(0);   gnssMsg.northVel_std = velo_std(1);   gnssMsg.upVel_std = velo_std(2);
    pubGNSS.publish(gnssMsg);

    Cluster_points.clear();
    clusteringCornerPointCloud(cornerPointsLessSharp);

    clusteringSurfPointCloud(surfPointsLessFlat);

    sensor_msgs::PointCloud2 laserCloudInMsg;
    pcl::toROSMsg(laserCloudIn, laserCloudInMsg);
    laserCloudInMsg.header.stamp = laserCloudMsg->header.stamp;
    laserCloudInMsg.header.frame_id = "/body";
    pubLaserCloudIn.publish(laserCloudInMsg);

    sensor_msgs::PointCloud2 laserCloudOutMsg;
    pcl::toROSMsg(*laserCloud, laserCloudOutMsg);
    laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
    laserCloudOutMsg.header.frame_id = "/body";
    pubLaserCloud.publish(laserCloudOutMsg);

    sensor_msgs::PointCloud2 cornerPointsSharpMsg;
    pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
    cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsSharpMsg.header.frame_id = "/body";
    pubCornerPointsSharp.publish(cornerPointsSharpMsg);

    sensor_msgs::PointCloud2 surfPointsFlat2;
    pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
    surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
    surfPointsFlat2.header.frame_id = "/body";
    pubSurfPointsFlat.publish(surfPointsFlat2);

    sensor_msgs::PointCloud2 clusterPointsMsg2;
    pcl::toROSMsg(Cluster_points, clusterPointsMsg2);
    clusterPointsMsg2.header.stamp = laserCloudMsg->header.stamp;
    clusterPointsMsg2.header.frame_id = "/body";
    pubClusterPoints.publish(clusterPointsMsg2);

    // pub each scam
    if(PUB_EACH_LINE)
    {
        for(int i = 0; i< N_SCANS; i++)
        {
            sensor_msgs::PointCloud2 scanMsg;
            pcl::toROSMsg(laserCloudScans[i], scanMsg);
            scanMsg.header.stamp = laserCloudMsg->header.stamp;
            scanMsg.header.frame_id = "/body";
            pubEachScan[i].publish(scanMsg);
        }
    }

    if(t_whole.toc() > 100)
        ROS_WARN("scan registration process over 100ms");

    double wholeProcessTime = t_whole.toc();
    ProcessTimeMean = ProcessTimeMean*FrameNum + wholeProcessTime;
    FrameNum++;
    ProcessTimeMean /= FrameNum;

    std_msgs::Float32 float_time;
    float_time.data = (float)ProcessTimeMean;
    pubProcessTime.publish(float_time);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "scanRegistration");
    ros::NodeHandle nh;

    std::string LidarTopic;
    nh.param<std::string>("lidar_topic", LidarTopic, "/velodyne_points");
    nh.param<int>("scan_line", N_SCANS, 32);
    nh.param<std::string>("lidar_type", LIDAR_TYPE, "HDL32");
    nh.param<double>("minimum_range", MINIMUM_RANGE, 0.1);
    nh.param<double>("clustering_size", clustering_size, 3);

    ROS_INFO("scan line number %d \n", N_SCANS);

    if(N_SCANS != 16 && N_SCANS != 32 && N_SCANS != 64)
    {
        ROS_ERROR("only support velodyne with 16, 32 or 64 scan line!");
        return 0;
    }

    if (LIDAR_TYPE == "VLP16" && N_SCANS == 16)
    {
        SharpEdgeNum = 2;
        LessSharpEdgeNum = 10;
        FlatSurfNum = 10;
        LessFlatSurfNum = 300;
    }
    else if (LIDAR_TYPE == "HDL32" && N_SCANS == 32)
    {
        SharpEdgeNum = 2;
        LessSharpEdgeNum = 10;
        FlatSurfNum = 10;
        LessFlatSurfNum = 200;
    }
    else if (LIDAR_TYPE == "HDL64" && N_SCANS == 64)
    {
        SharpEdgeNum = 1;
        LessSharpEdgeNum = 5;
        FlatSurfNum = 5;
        LessFlatSurfNum = 80;
    }

    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(LidarTopic, 100, laserCloudHandler);
    ros::Subscriber subGNSSINS = nh.subscribe<novatel_gps_msgs::Inspvax>("/inspvax", 100, gnssHandler);

    pubLaserCloudIn = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_1", 100);

    pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100);

    pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("/feature/laser_cloud_sharp", 100);

    pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("/feature/laser_cloud_flat", 100);

    pubRemovePoints = nh.advertise<sensor_msgs::PointCloud2>("/laser_remove_points", 100);

    pubClusterPoints = nh.advertise<sensor_msgs::PointCloud2>("/feature/laser_feature_cluster_points", 100);

    pubGNSS = nh.advertise<pd_loam::gnss>("/GNSS", 100);

    pubProcessTime = nh.advertise<std_msgs::Float32>("/feature_process_time", 100);

    pubgnssText = nh.advertise<jsk_rviz_plugins::OverlayText>("/gnss_text", 100);

    if(PUB_EACH_LINE)
    {
        for(int i = 0; i < N_SCANS; i++)
        {
            ros::Publisher tmp = nh.advertise<sensor_msgs::PointCloud2>("/laser_scanid_" + std::to_string(i), 100);
            pubEachScan.push_back(tmp);
        }
    }
    ros::spin();

    return 0;
}
