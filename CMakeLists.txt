# set(CMAKE_BUILD_TYPE Release)
cmake_minimum_required(VERSION 3.0.2)
project(pcmf)

# Compile as C++14, supported in ROS Kinetic and newer
add_compile_options(-std=c++17)

# Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  pcl_ros
  sensor_msgs
  visualization_msgs
  # ndt_omp  # Add ndt_omp here
)

# System dependencies are found with CMake's conventions
find_package(Eigen3 REQUIRED)

set(PCL_DIR "/usr/lib/x86_64-linux-gnu/cmake/pcl/PCLConfig.cmake/")
find_package(PCL 1.7 REQUIRED COMPONENTS common io features keypoints registration)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
find_package(VTK REQUIRED)
include_directories(${VTK_INCLUDE_DIRS})


# add_definitions(-msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2)
# set(CMAKE_CXX_FLAGS "-msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2")

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp pcl_ros sensor_msgs visualization_msgs
  DEPENDS EIGEN3 PCL
)

###########
## Build ##
###########

# Specify additional locations of header files
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${ndt_omp_INCLUDE_DIRS}
)

# Declare a C++ executable for map_processing_node
add_executable(${PROJECT_NAME}_map_processing_node src/map_processing_node.cpp)
add_dependencies(${PROJECT_NAME}_map_processing_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_map_processing_node
  ${catkin_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  ${PCL_LIBRARIES}
)

# # Declare a C++ executable for icp_pcmf_node
# add_executable(${PROJECT_NAME}_icp_pcmf_node src/icp_pcmf_node.cpp)
# add_dependencies(${PROJECT_NAME}_icp_pcmf_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# target_link_libraries(${PROJECT_NAME}_icp_pcmf_node
#   ${catkin_LIBRARIES}
#   ${EIGEN3_LIBRARIES}
#   ${PCL_LIBRARIES}
#   ${VTK_LIBRARIES}
# )

# Declare a C++ executable for anchor_pivot_node
add_executable(${PROJECT_NAME}_anchor_pivot_node src/anchor_pivot_node.cpp)
add_dependencies(${PROJECT_NAME}_anchor_pivot_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_anchor_pivot_node
  ${catkin_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  ${PCL_LIBRARIES}
  ${VTK_LIBRARIES}
)

# Mark the npy_map_publisher.py script for installation
catkin_install_python(PROGRAMS scripts/npy_map_publisher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Mark the bag_player.py script for installation
catkin_install_python(PROGRAMS scripts/bag_player.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Mark the corner_heatmap_node.py script for installation
catkin_install_python(PROGRAMS scripts/corner_heatmap_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Mark the ekf_transformation_plotter.py script for installation
catkin_install_python(PROGRAMS scripts/ekf_transformation_plotter.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Mark the gps_to_local_frame.py script for installation
catkin_install_python(PROGRAMS scripts/gps_to_local_frame.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Mark the gps_slam_error_plotter.py script for installation
catkin_install_python(PROGRAMS scripts/gps_slam_error_plotter.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Mark the dxf_map_publisher.py script for installation
catkin_install_python(PROGRAMS scripts/predef_map_publisher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Mark the bridge_gps_to_local_frame.py script for installation
catkin_install_python(PROGRAMS scripts/bridge_gps_to_local_frame.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Mark the topic_logger_to_csv.py script for installation
catkin_install_python(PROGRAMS plot_scripts/topic_logger_to_csv.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Mark the predef_map_publisher_brattora.py script for installation
catkin_install_python(PROGRAMS scripts/predef_map_publisher_brattora.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)