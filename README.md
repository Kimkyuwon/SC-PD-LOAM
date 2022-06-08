# What is SC-PD-LOAM?
* SC-PD-LOAM is Probability Distribution based LOAM and Pose Graph Optimization method.
* Scan matching is performed using probability distributions of edge and planar points.
* Loop detection for pose graph optimization is performed using scan-context.
![node_graph](https://user-images.githubusercontent.com/5857457/172543497-abd3893f-8ce0-4158-a4be-76f990031d54.png)
![ezgif com-gif-maker](https://user-images.githubusercontent.com/5857457/172544889-da00c369-3c58-44b2-85b2-8cd69a319f58.gif)
# Requirements
* Cmake
* PCL-1.8
* Eigen3
* GTSAM
* Ceres-solver
* jsk-rviz-plugins (optional)
# How to build
```
$ cd ~/YOUR_WORKSPACE/src/
$ git clone https://github.com/Kimkyuwon/SC-PD-LOAM
$ cd ..
$ catkin_make
$ source ~/YOUR_WORKSPACE/devel/setup.bash
```
# Contact
If you have any questions, contact here please
```
kkw1125@konkuk.ac.kr
```
