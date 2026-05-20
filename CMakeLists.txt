cmake_minimum_required(VERSION 3.16)
project(microamp_fgo_rov_tracking)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── ROS dependencies ─────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(blueboat_interfaces REQUIRED)

# ── Third-party ───────────────────────────────────────────────────────────────
find_package(GTSAM REQUIRED)
find_package(Eigen3 REQUIRED)

# ── Target ────────────────────────────────────────────────────────────────────
add_executable(fgo_tracking_node 
  src/fgo_tracking_node.cpp
  src/config.cpp
  src/rov_factors.cpp
)

# ament_target_dependencies uses CMake's plain (non-keyword) signature
# internally. All target_link_libraries calls on the same target must use the
# same style — so we also use the plain signature for gtsam/Eigen below.
ament_target_dependencies(fgo_tracking_node
  rclcpp
  sensor_msgs
  geometry_msgs
  nav_msgs
  tf2_ros
  blueboat_interfaces
)

target_include_directories(fgo_tracking_node PRIVATE
  include
  ${GTSAM_INCLUDE_DIR}    # e.g. /usr/local/include
  ${EIGEN3_INCLUDE_DIR}   # e.g. /usr/include/eigen3
)

# Plain signature — must match ament_target_dependencies above
target_link_libraries(fgo_tracking_node
  gtsam
  Eigen3::Eigen
)

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS fgo_tracking_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)


install(DIRECTORY include/
  DESTINATION include
)

# ── Tests ─────────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)

  # --- USBL Jacobian test ----------------------------------------------------
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_usbl_jacobian
    test/test_usbl_jacobian.cpp
    src/rov_factors.cpp
  )

  target_include_directories(test_usbl_jacobian PRIVATE
    include
    ${GTSAM_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
  )

  target_link_libraries(test_usbl_jacobian
    gtsam
    Eigen3::Eigen
  )

  ament_target_dependencies(test_usbl_jacobian
    rclcpp
    sensor_msgs
    geometry_msgs
    nav_msgs
    tf2_ros
    blueboat_interfaces
  )
endif()


ament_package()