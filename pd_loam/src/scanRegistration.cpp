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

#define DEBUG_MODE_FEATURE 0

#include <cmath>
#include <vector>
#include <string>
#include "common.h"
#include "tic_toc.h"
#include <nav_msgs/Odometry.h>
#include <opencv/cv.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/search.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/normal_3d.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float32.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

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
float cloudCurvature[400000];
int cloudSortInd[400000];
int cloudNeighborPicked[400000];
int cloudLabel[400000];

int FrameNum = 0;
double ProcessTimeMean = 0;

bool comp (int i,int j) { return (cloudCurvature[i]<cloudCurvature[j]); }

ros::Publisher pubLaserCloud;
ros::Publisher pubCornerPointsSharp;
ros::Publisher pubSurfPointsFlat;
ros::Publisher pubRemovePoints;
ros::Publisher pubClusterPoints;
ros::Publisher pubProcessTime;
std::vector<ros::Publisher> pubEachScan;

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

void clusteringCornerPointCloud(const pcl::PointCloud<PointType> &cornerPointsLessSharp, std::vector<int> clusterPicked,
                                std::vector<pcl::PointCloud<PointType>> &corner_clusterPC)
{
    pcl::PointCloud<PointType> cluster_points;
    for (size_t i = 0; i < cornerPointsLessSharp.size() - 1; i++)
    {
        if (clusterPicked[i] == 1)  continue;

        cluster_points.clear();
        PointType t;
        t = cornerPointsLessSharp[i];
        cluster_points.push_back(t);
        clusterPicked[i] = 1;
        float min_dist = 2;
        int currentInd = -1;
        int compLayer = (int)cornerPointsLessSharp[i]._PointXYZINormal::curvature;
        int currentLayer = (int)cornerPointsLessSharp[i]._PointXYZINormal::curvature;
        for (size_t j = i + 1; j < cornerPointsLessSharp.size(); j++)
        {
            if ((int)cornerPointsLessSharp[i]._PointXYZINormal::curvature == (int)cornerPointsLessSharp[j]._PointXYZINormal::curvature)
            {
                continue;
            }
            compLayer = (int)cornerPointsLessSharp[j]._PointXYZINormal::curvature;

            if (currentLayer != compLayer && currentInd != -1 && clusterPicked[currentInd] != 1 )
            {
                PointType t;
                t = cornerPointsLessSharp[currentInd];
                cluster_points.push_back(t);
                clusterPicked[currentInd] = 1;
                currentInd = -1;
                min_dist = 2;
            }

            PointType temp_curr_point = cornerPointsLessSharp[i];
            PointType temp_comp_point = cornerPointsLessSharp[j];
            float diffX = temp_curr_point.x - temp_comp_point.x;
            float diffY = temp_curr_point.y - temp_comp_point.y;
            float diffDist = sqrt(diffX * diffX + diffY * diffY);

            if (diffDist > min_dist)   continue;
            min_dist = diffDist;
            currentInd = j;
            currentLayer = (int)cornerPointsLessSharp[j]._PointXYZINormal::curvature;
        }
        if (cluster_points.size() >= 5)
        {
            corner_clusterPC.push_back(cluster_points);
        }
    }
}

void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
    if (!systemInited)
    {
        systemInitCount++;
        if (systemInitCount >= systemDelay)
        {
            systemInited = true;
        }
        else
            return;
    }

    TicToc t_whole;
    TicToc t_feature;
    TicToc t_prepare;
    std::vector<int> scanStartInd(N_SCANS, 0);
    std::vector<int> scanEndInd(N_SCANS, 0);

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

            // use [0 50]  > 50 remove outlies
            if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0)
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
    }

    //printf("prepare time %f \n", t_prepare.toc());

    for (int i = 5; i < cloudSize - 5; i++)
    {
        float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x + laserCloud->points[i + 3].x + laserCloud->points[i + 4].x + laserCloud->points[i + 5].x;
        float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y + laserCloud->points[i + 3].y + laserCloud->points[i + 4].y + laserCloud->points[i + 5].y;
        float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z + laserCloud->points[i + 3].z + laserCloud->points[i + 4].z + laserCloud->points[i + 5].z;

        cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;
        cloudSortInd[i] = i;
        cloudNeighborPicked[i] = 0;
        cloudLabel[i] = 0;
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
                    cloudCurvature[ind] > CURVATURE_THRESHOLD)
                {

                    largestPickedNum++;
                    if (largestPickedNum <= 2)
                    {
                        cloudLabel[ind] = 2;
                        cornerPointsSharp.push_back(laserCloud->points[ind]);
                        cornerPointsLessSharpScan->push_back(laserCloud->points[ind]);
                    }
                    else if (largestPickedNum <= 10)
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
                    cloudCurvature[ind] < CURVATURE_THRESHOLD)
                {

                    cloudLabel[ind] = -1;
                    surfPointsFlat.push_back(laserCloud->points[ind]);

                    smallestPickedNum++;
                    if (smallestPickedNum >= 5)
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

            for (int k = sp; k <= ep; k++)
            {
                if (cloudLabel[k] <= 0)
                {
                    surfPointsLessFlatScan->push_back(laserCloud->points[k]);
                }
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
    printf("feature extraction time %f ms *************\n", t_feature.toc());
#endif

    pcl::PointCloud<PointType>::Ptr surfPointsLessFlatPtr (new pcl::PointCloud<PointType> (surfPointsLessFlat));
    pcl::PointCloud<PointType>::Ptr cornerPointsLessSharpPtr (new pcl::PointCloud<PointType> (cornerPointsLessSharp));

    std::vector<pcl::PointCloud<PointType>> corner_clusterPC;
    pcl::PointCloud<PointType> cluster_points;
    std::vector<int> clusterPicked(cornerPointsLessSharp.size(), 0);

    clusteringCornerPointCloud(cornerPointsLessSharp, clusterPicked, corner_clusterPC);

    pcl::PointCloud<pcl::PointXYZINormal> Cluster_points;
    int cnt = 0;
    for (size_t k = 0; k < corner_clusterPC.size(); k++)
    {
        for (size_t ii = 0; ii < corner_clusterPC[k].size(); ii++)
        {
            pcl::PointXYZINormal p;
            p.x = corner_clusterPC[k].points[ii].x;
            p.y = corner_clusterPC[k].points[ii].y;
            p.z = corner_clusterPC[k].points[ii].z;
            p.intensity = corner_clusterPC[k].points[ii].intensity;
            p._PointXYZINormal::curvature = corner_clusterPC[k].points[ii]._PointXYZINormal::curvature; //layer and scan time
            p._PointXYZINormal::normal_y = cnt;  //cluster num
            p._PointXYZINormal::normal_z = 1;    //feature point type (1: corner, 2: surf)
            Cluster_points.push_back(p);
        }
        cnt = cnt + 1;
    }

    pcl::PointCloud<PointType> tmp_cluster_points;
    std::vector<int> tmpclusterPicked(surfPointsLessFlatPtr->points.size(), 0);
    cnt = 0;
    for (size_t kk = 0; kk < surfPointsLessFlatPtr->points.size(); kk++)
    {
        if (tmpclusterPicked[kk] == 1)  continue;

        tmp_cluster_points.clear();
        PointType t;
        t = surfPointsLessFlatPtr->points[kk];
        tmp_cluster_points.push_back(t);
        tmpclusterPicked[kk] = 1;
        float dist_Threshold = 3;
        for (size_t jj = kk + 1; jj < surfPointsLessFlatPtr->points.size(); jj++)
        {
            PointType temp_curr_point = surfPointsLessFlatPtr->points[kk];
            PointType temp_comp_point = surfPointsLessFlatPtr->points[jj];
            float diffX = temp_curr_point.x - temp_comp_point.x;
            float diffY = temp_curr_point.y - temp_comp_point.y;
            float diffZ = temp_curr_point.z - temp_comp_point.z;

            float diffDist = sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ);

            if (diffDist < dist_Threshold && tmpclusterPicked[jj] != 1)
            {
                tmpclusterPicked[jj] = 1;
                tmp_cluster_points.push_back(surfPointsLessFlatPtr->points[jj]);
            }
        }

        if (tmp_cluster_points.size() >= 5)
        {
            for (size_t ii = 0; ii < tmp_cluster_points.size(); ii++)
            {
                pcl::PointXYZINormal p;
                p.x = tmp_cluster_points.points[ii].x;
                p.y = tmp_cluster_points.points[ii].y;
                p.z = tmp_cluster_points.points[ii].z;
                p.intensity = tmp_cluster_points.points[ii].intensity;
                p._PointXYZINormal::curvature = tmp_cluster_points.points[ii]._PointXYZINormal::curvature; //layer and scan time
                p._PointXYZINormal::normal_y = cnt;  //cluster num
                p._PointXYZINormal::normal_z = 2;    //feature point type (1: corner, 2: surf)
                Cluster_points.push_back(p);
            }
            cnt = cnt + 1;
        }
    }

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

#if DEBUG_MODE_FEATURE == 1
    printf("scan registration time %f ms *************\n", t_whole.toc());
#endif
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

    nh.param<int>("scan_line", N_SCANS, 32);
    nh.param<std::string>("lidar_type", LIDAR_TYPE, "HDL32");
    nh.param<double>("minimum_range", MINIMUM_RANGE, 0.1);

    ROS_INFO("scan line number %d \n", N_SCANS);

    if(N_SCANS != 16 && N_SCANS != 32 && N_SCANS != 64)
    {
        ROS_ERROR("only support velodyne with 16, 32 or 64 scan line!");
        return 0;
    }

    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, laserCloudHandler);

    pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100);

    pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("/feature/laser_cloud_sharp", 100);

    pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("/feature/laser_cloud_flat", 100);

    pubRemovePoints = nh.advertise<sensor_msgs::PointCloud2>("/laser_remove_points", 100);

    pubClusterPoints = nh.advertise<sensor_msgs::PointCloud2>("/feature/laser_feature_cluster_points", 100);

    pubProcessTime = nh.advertise<std_msgs::Float32>("/feature_process_time", 100);

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
