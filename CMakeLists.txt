cmake_minimum_required(VERSION 3.0.2)
project(pd_loam)

set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  rosbag
  std_msgs  
  message_generation
  image_transport
  cv_bridge
  tf
)

#find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

find_package(OpenMP REQUIRED)
find_package(GTSAM REQUIRED QUIET)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
)

add_message_files(
FILES
frame.msg
)

generate_messages(
DEPENDENCIES
std_msgs
nav_msgs
sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime geometry_msgs nav_msgs roscpp rospy std_msgs
  DEPENDS EIGEN3 PCL
  INCLUDE_DIRS include
)


add_executable(scanRegistration src/scanRegistration.cpp)
target_link_libraries(scanRegistration ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(laserOdometry src/laserOdometry.cpp)
target_link_libraries(laserOdometry ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES})

add_executable(laserMapping src/laserMapping.cpp)
target_link_libraries(laserMapping ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES} ${OpenMP_CXX_FLAGS} gtsam)
add_dependencies(laserMapping ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(imuPreIntegration src/imuPreIntegration.cpp)
target_link_libraries(imuPreIntegration ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES} gtsam)

add_executable(imuPreIntegration_kitti src/imuPreIntegration_kitti.cpp)
target_link_libraries(imuPreIntegration_kitti ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES} gtsam)

add_executable(mapVisualization src/mapVisualization.cpp)
target_link_libraries(mapVisualization ${catkin_LIBRARIES} ${PCL_LIBRARIES})
add_dependencies(mapVisualization ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(laserPG
    src/laserPoseGraph.cpp
    include/scancontext/Scancontext.cpp)
#target_compile_options(${PROJECT_NAME}_laserPG
#  PRIVATE ${OpenMP_CXX_FLAGS}
#)
target_link_libraries(laserPG
 ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES}
  ${OpenMP_CXX_FLAGS}
  gtsam)
add_dependencies(laserPG ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
