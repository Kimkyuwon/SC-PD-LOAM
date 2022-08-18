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

#pragma once

#include <cmath>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <visualization_msgs/MarkerArray.h>

typedef pcl::PointXYZINormal PointType;

inline double rad2deg(double radians)
{
  return radians * 180.0 / M_PI;
}

inline double deg2rad(double degrees)
{
  return degrees * M_PI / 180.0;
}


// Function to find mean.
Eigen::Vector3d getMean(pcl::PointCloud<PointType> points)
{
    Eigen::Vector3d sum;
    sum.setZero();
    for (size_t i = 0; i < points.size(); i++)
    {
        sum(0) = sum(0) + points.points[i].x;
        sum(1) = sum(1) + points.points[i].y;
        sum(2) = sum(2) + points.points[i].z;
    }
    sum /= points.size();
    return sum;
}

// Function to find covariance.
Eigen::Matrix3d getCovariance(pcl::PointCloud<PointType> points)
{
    Eigen::Vector3d meanPoint = getMean(points);
    Eigen::Matrix3d cov;
    cov.setZero();
    for (size_t i = 0; i < points.size(); i++)
    {
        cov(0,0) += (points.points[i].x - meanPoint(0))*(points.points[i].x - meanPoint(0));
        cov(0,1) += (points.points[i].x - meanPoint(0))*(points.points[i].y - meanPoint(1));
        cov(0,2) += (points.points[i].x - meanPoint(0))*(points.points[i].z - meanPoint(2));
        cov(1,0) += (points.points[i].y - meanPoint(1))*(points.points[i].x - meanPoint(0));
        cov(1,1) += (points.points[i].y - meanPoint(1))*(points.points[i].y - meanPoint(1));
        cov(1,2) += (points.points[i].y - meanPoint(1))*(points.points[i].z - meanPoint(2));
        cov(2,0) += (points.points[i].z - meanPoint(2))*(points.points[i].x - meanPoint(0));
        cov(2,1) += (points.points[i].z - meanPoint(2))*(points.points[i].y - meanPoint(1));
        cov(2,2) += (points.points[i].z - meanPoint(2))*(points.points[i].z - meanPoint(2));
    }
    return cov / (points.size() - 1);
}

Eigen::Vector3d llh2xyz(Eigen::Vector3d gps_llh)
{
    double phi = gps_llh(0) * M_PI/180;
    double lambda = gps_llh(1) * M_PI/180;
    double h = gps_llh(2);

    double a = 6378137.000;
    double b = 6356752.3142;
    double e = sqrt(1-pow((b/a),2));

    double sinphi = sin(phi);
    double cosphi = cos(phi);
    double coslam = cos(lambda);
    double sinlam = sin(lambda);
    double tan2phi = pow(tan(phi),2);
    double tmp = 1 - pow(e,2);
    double tmpden = sqrt(1+tmp*tan2phi);

    double x = (a*coslam)/tmpden + h*coslam*cosphi;
    double y = (a*sinlam)/tmpden + h*sinlam*cosphi;

    double tmp2 = sqrt(1-pow(e,2)*pow(sinphi,2));
    double z = (a*tmp*sinphi)/tmp2 + h*sinphi;

    Eigen::Vector3d xyz_result;
    xyz_result(0) = x;  xyz_result(1) = y;  xyz_result(2) = z;
    return xyz_result;
}

Eigen::Vector3d xyz2llh (Eigen::Vector3d gps_xyz)
{
    double x = gps_xyz(0);
    double y = gps_xyz(1);
    double z = gps_xyz(2);
    double x2 = pow(x,2);   double y2 = pow(y,2);   double z2 = pow(z,2);
    double a = 6378137.000;
    double b = 6356752.3142;
    double e = sqrt(1-pow((b/a),2));
    double b2 = pow(b,2);   double e2 = pow(e,2);
    double ep = e*(a/b);
    double r = sqrt(x2+y2);
    double r2 = pow(r,2);
    double E2 = pow(a,2) - pow(b,2);
    double F = 54*b2*z2;
    double G = r2+(1-e2)*z2 - e2*E2;
    double c = (pow(e2,2)*F*r2)/(pow(G,3));
    double third = 1/3;
    double s = pow((1+c+sqrt(pow(c,2)+2*c)),third);
    double P = F/(3*pow((s+1/s+1),2)*pow(G,2));
    double Q = sqrt(1+2*e2*e2*P);
    double ro = -(P*e2*r)/(1+Q) + sqrt((a*a/2)*(1+1/Q) - (P*(1-e2)*z2)/(Q*(1+Q)) - P*r2/2);

    double tmp = (r - e2*ro)*(r - e2*ro);
    double U = sqrt(tmp+z2);
    double V = sqrt(tmp + (1-e2)*z2);
    double zo = (b2*z)/(a*V);
    double height = U*(1-b2/(a*V));

    double lat = atan((z+ep*ep*zo)/r);
    double temp = atan(y/x);

    double longi;
    if (x>=0)
    {
        longi = temp;
    }
    else if (x < 0 && y >= 0)
    {
        longi = M_PI + temp;
    }
    else
    {
        longi = temp - M_PI;
    }
    Eigen::Vector3d llh_result;
    llh_result(0) = lat * 180/M_PI;
    llh_result(1) = longi * 180/M_PI;
    llh_result(2) = height;

    return llh_result;
}

Eigen::Vector3d xyz2enu(Eigen::Vector3d gps_xyz, Eigen::Vector3d origin_xyz)
{
    Eigen::Vector3d tmpxyz = gps_xyz;
    Eigen::Vector3d tmporg = origin_xyz;
    Eigen::Vector3d diff_xyz;
    diff_xyz(0) = tmpxyz(0) - tmporg(0);
    diff_xyz(1) = tmpxyz(1) - tmporg(1);
    diff_xyz(2) = tmpxyz(2) - tmporg(2);

    Eigen::Vector3d origllh;
    origllh = xyz2llh(origin_xyz);
    double phi = origllh(0) * M_PI/180;
    double lam = origllh(1) * M_PI/180;
    double sinphi = sin(phi);
    double cosphi = cos(phi);
    double sinlam = sin(lam);
    double coslam = cos(lam);

    Eigen::Matrix3d R;
    R(0,0) = -sinlam;   R(0,1) = coslam;    R(0,2) = 0;
    R(1,0) = -sinphi*coslam;    R(1,1) = -sinphi*sinlam;    R(1,2) = cosphi;
    R(2,0) = cosphi*coslam; R(2,1) = cosphi*sinlam; R(2,2) = sinphi;

    Eigen::Vector3d result_enu = R*diff_xyz;
    return result_enu;
}

Eigen::Vector3d enu2xyz(Eigen::Vector3d gps_llh, Eigen::Vector3d gps_enu)
{
    Eigen::Vector3d refllh;
    refllh(0) = gps_llh(0)*M_PI/180;
    refllh(1) = gps_llh(1)*M_PI/180;
    refllh(2) = gps_llh(2);
    Eigen::Vector3d refxyz = llh2xyz(gps_llh);

    Eigen::Vector3d result_xyz;
    result_xyz(0) = -sin(refllh(1))*gps_enu(0) - cos(refllh(1))*sin(refllh(0))*gps_enu(1) + cos(refllh(1))*cos(refllh(0))*gps_enu(2) + refxyz(0);
    result_xyz(1) = cos(refllh(1))*gps_enu(0) - sin(refllh(1))*sin(refllh(0))*gps_enu(1) + cos(refllh(0))*sin(refllh(1))*gps_enu(2) + refxyz(1);
    result_xyz(2) = cos(refllh(0))*gps_enu(1) + sin(refllh(0))*gps_enu(2) + refxyz(2);
    return result_xyz;
}

struct Pose6D {
  double x;
  double y;
  double z;
  double roll;
  double pitch;
  double yaw;
};
