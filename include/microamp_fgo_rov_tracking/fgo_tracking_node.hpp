#ifndef MICROAMP_FACTOR_GRAPH_TRACKING_NODE_HPP
#define MICROAMP_FACTOR_GRAPH_TRACKING_NODE_HPP

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <memory>
#include <mutex>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"

// Custom message types (replace with your actual message packages)
#include "blueboat_interfaces/msg/gnss_nav_pvt.hpp"
#include "blueboat_interfaces/msg/boat_state.hpp"
#include "blueboat_interfaces/msg/usbl_measurement.hpp"
#include "blueboat_interfaces/msg/acoustic_range.hpp"
#include "blueboat_interfaces/msg/rov_depth.hpp"
#include "blueboat_interfaces/msg/rov_state.hpp"

// NED conversion utility (your existing utility)
#include "ned_converter.hpp"

using gtsam::symbol_shorthand::B; // Bias  (b)
using gtsam::symbol_shorthand::V; // Velocity (v)
using gtsam::symbol_shorthand::X; // Pose (x)

// New symbols for ROV
namespace gtsam {
namespace symbol_shorthand {
inline Key R(std::uint64_t j) { return Symbol('r', j); } // ROV position
inline Key W(std::uint64_t j) { return Symbol('w', j); } // ROV velocity
} // namespace symbol_shorthand
} // namespace gtsam

using gtsam::symbol_shorthand::R; // ROV position
using gtsam::symbol_shorthand::W; // ROV velocity

class FactorGraphTrackingNode : public rclcpp::Node {
public:
  using Imu = sensor_msgs::msg::Imu;
  using GNSSNavPvt = blueboat_interfaces::msg::GNSSNavPvt;
  using BoatState = blueboat_interfaces::msg::BoatState;
  using USBLMeasurement = blueboat_interfaces::msg::USBLMeasurement;
  using AcousticRange = blueboat_interfaces::msg::AcousticRange;
  using ROVDepth = blueboat_interfaces::msg::ROVDepth;
  using ROVState = blueboat_interfaces::msg::ROVState;

  FactorGraphTrackingNode();

private:
  // ==================== ASV STATE ====================
  void imuCallback(const Imu::SharedPtr msg);
  void gnssCallback(const GNSSNavPvt::SharedPtr msg);
  void initializeDatumFromGNSS(const GNSSNavPvt &msg);
  void initializeGraph();
  void publishBoatState(const gtsam::Values &est);

  // ==================== ROV STATE ====================
  void usblCallback(const USBLMeasurement::SharedPtr msg);
  void rangeCallback(const AcousticRange::SharedPtr msg);
  void depthCallback(const ROVDepth::SharedPtr msg);
  void initializeROVState();
  void publishROVState(const gtsam::Values &est);

  // ==================== SHARED ====================
  gtsam::Values updateAndGetEstimate();

  // ==================== GTSAM CORE ====================
  gtsam::ISAM2 isam2_;
  gtsam::NonlinearFactorGraph graph_;
  gtsam::Values values_;
  std::mutex graph_mutex_;

  // ==================== ASV NAVIGATION ====================
  std::unique_ptr<gtsam::PreintegratedCombinedMeasurements> pim_;
  gtsam::imuBias::ConstantBias bias_;
  double gravity_;

  NEDConverter ned_;
  bool datum_initialised_;
  bool graph_initialised_;
  bool imu_initialised_ = false;

  uint64_t key_; // Shared timestep counter for both ASV and ROV

  rclcpp::Time last_imu_time_;
  double last_gyro_z_ = 0.0;

  // ==================== ROV STATE ====================
  bool rov_initialised_;
  rclcpp::Time last_rov_update_time_;

  // ROV sensor buffers (for temporal alignment)
  struct USBLBuffer {
    rclcpp::Time stamp;
    double azimuth;
    double elevation;
    double azimuth_std;
    double elevation_std;
  };
  std::optional<USBLBuffer> pending_usbl_;

  struct RangeBuffer {
    rclcpp::Time stamp;
    double range;
    double range_std;
  };
  std::optional<RangeBuffer> pending_range_;

  struct DepthBuffer {
    rclcpp::Time stamp;
    double depth;
    double depth_std;
  };
  std::optional<DepthBuffer> pending_depth_;

  // ==================== PARAMETERS ====================
  // ASV IMU/GNSS
  double accel_noise_, gyro_noise_;
  double accel_rw_, gyro_rw_;
  double prior_pose_sigma_, prior_vel_sigma_, prior_bias_sigma_;
  double gps_sigma_floor_, gps_sigma_max_;

  // ROV
  double rov_prior_pos_sigma_;
  double rov_prior_vel_sigma_;
  double rov_process_vel_sigma_; // Constant velocity model noise
  double usbl_azimuth_sigma_;
  double usbl_elevation_sigma_;
  double acoustic_range_sigma_;
  double rov_depth_sigma_;

  // USBL mounting offset (relative to ASV IMU frame)
  gtsam::Point3 usbl_offset_; // (x, y, z) in ASV body frame

  // ==================== ROS SUBSCRIPTIONS ====================
  rclcpp::Subscription<Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<GNSSNavPvt>::SharedPtr gnss_sub_;
  rclcpp::Subscription<USBLMeasurement>::SharedPtr usbl_sub_;
  rclcpp::Subscription<AcousticRange>::SharedPtr range_sub_;
  rclcpp::Subscription<ROVDepth>::SharedPtr depth_sub_;

  // ==================== ROS PUBLISHERS ====================
  rclcpp::Publisher<BoatState>::SharedPtr state_pub_;
  rclcpp::Publisher<ROVState>::SharedPtr rov_state_pub_;
};

#endif // MICROAMP_FACTOR_GRAPH_TRACKING_NODE_HPP
