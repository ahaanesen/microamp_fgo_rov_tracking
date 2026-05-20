// Standalone test for UsblFactor Jacobian verification.
//
// Build (adjust paths to your GTSAM install):
//   g++ -std=c++17 -O2 test_usbl_jacobian.cpp rov_factors.cpp \
//       -I/usr/local/include -L/usr/local/lib -lgtsam -lboost_serialization \
//       -o test_usbl_jacobian
//
// Or wire it into your CMake as a separate ament_add_gtest / catkin test target.

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/base/numericalDerivative.h>
#include "microamp_fgo_rov_tracking/rov_factors.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>

using namespace gtsam;

// Wrap evaluateError into a std::function for numericalDerivative.
// Note: numericalDerivative does not pass H matrices, so we evaluate without them.
static Vector usblErrorWrapper(
    const UsblFactor& factor,
    const Pose3& asvPose,
    const Point3& rovPoint)
{
    return factor.evaluateError(asvPose, rovPoint);
}

struct TestCase {
    std::string name;
    Pose3 asvPose;
    Point3 rovPoint;
    double measAzDeg;
    double measElDeg;
};

bool runCase(const TestCase& tc, const Pose3& body_P_sensor, double tol = 1e-5) {
    // Identity-noise model (factor returns raw, unwhitened error here).
    auto noise = noiseModel::Isotropic::Sigma(2, 1.0);
    UsblFactor factor(0, 1, tc.measAzDeg, tc.measElDeg, body_P_sensor, noise);

    // Analytical Jacobians from the factor itself.
    Matrix H_asv_analytical, H_rov_analytical;
    Vector e = factor.evaluateError(
        tc.asvPose, tc.rovPoint,
        &H_asv_analytical, &H_rov_analytical);

    // Numerical Jacobians via GTSAM utility.
    Matrix H_asv_numerical = numericalDerivative21<Vector, Pose3, Point3>(
        [&](const Pose3& p, const Point3& r) { return usblErrorWrapper(factor, p, r); },
        tc.asvPose, tc.rovPoint, 1e-6);

    Matrix H_rov_numerical = numericalDerivative22<Vector, Pose3, Point3>(
        [&](const Pose3& p, const Point3& r) { return usblErrorWrapper(factor, p, r); },
        tc.asvPose, tc.rovPoint, 1e-6);

    double err_asv = (H_asv_analytical - H_asv_numerical).cwiseAbs().maxCoeff();
    double err_rov = (H_rov_analytical - H_rov_numerical).cwiseAbs().maxCoeff();

    bool pass = (err_asv < tol) && (err_rov < tol);

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "\n=== " << tc.name << " ===\n";
    std::cout << "  error vector:  " << e.transpose() << "\n";
    std::cout << "  max |H_asv_analytical - H_asv_numerical| = " << err_asv
              << (err_asv < tol ? "  OK" : "  FAIL") << "\n";
    std::cout << "  max |H_rov_analytical - H_rov_numerical| = " << err_rov
              << (err_rov < tol ? "  OK" : "  FAIL") << "\n";

    if (!pass) {
        std::cout << "  H_asv (analytical):\n" << H_asv_analytical << "\n";
        std::cout << "  H_asv (numerical):\n"  << H_asv_numerical  << "\n";
        std::cout << "  H_rov (analytical):\n" << H_rov_analytical << "\n";
        std::cout << "  H_rov (numerical):\n"  << H_rov_numerical  << "\n";
    }
    return pass;
}

int main() {
    // USBL lever arm: 1.2 m below body origin, no rotation (matches your config defaults).
    Pose3 body_P_sensor(Rot3::Identity(), Point3(0.0, 0.0, 1.2));

    // Build a few representative scenarios. NED frame: x=N, y=E, z=Down.
    std::vector<TestCase> cases;

    // Case 1: ROV directly to the East and below the ASV. Nominal geometry.
    cases.push_back({
        "Nominal: ROV East and below",
        Pose3(Rot3::Identity(), Point3(0, 0, 0)),
        Point3(0.0, 20.0, 15.0),
        90.0,   // expected az ~ 90 deg
        36.0    // expected el ~ atan2(13.8, 20) in deg
    });

    // Case 2: ROV to the North-East with ASV yawed 30 deg.
    cases.push_back({
        "Yawed ASV, ROV NE-below",
        Pose3(Rot3::Ypr(30.0 * M_PI/180.0, 0, 0), Point3(5, 5, 0)),
        Point3(20.0, 25.0, 30.0),
        45.0,
        45.0
    });

    // Case 3: ROV with non-trivial ASV roll and pitch.
    cases.push_back({
        "Tilted ASV (roll/pitch), ROV ahead",
        Pose3(Rot3::Ypr(0.1, 0.05, -0.08), Point3(-2, 3, 0)),
        Point3(30.0, 10.0, 25.0),
        20.0,
        40.0
    });

    // Case 4: ROV almost directly below (small horizontal distance) — stresses the
    // r_raw clamp. Both Jacobian rows should still match numerical.
    cases.push_back({
        "ROV nearly overhead (small horizontal)",
        Pose3(Rot3::Identity(), Point3(0, 0, 0)),
        Point3(0.05, 0.05, 25.0),
        45.0,
        88.0
    });

    // Case 5: ROV behind a yawed ASV (azimuth near 180 deg — wrap zone).
    cases.push_back({
        "Azimuth near wrap (180 deg)",
        Pose3(Rot3::Ypr(0.0, 0, 0), Point3(0, 0, 0)),
        Point3(-30.0, 0.5, 10.0),
        179.0,
        18.0
    });

    // Case 6: ROV at large horizontal distance (well-conditioned).
    cases.push_back({
        "Far ROV, mostly horizontal",
        Pose3(Rot3::Ypr(0.4, 0.02, -0.01), Point3(10, -5, 0)),
        Point3(100.0, -80.0, 50.0),
        310.0,
        20.0
    });

    int n_pass = 0, n_total = cases.size();
    for (const auto& tc : cases) {
        if (runCase(tc, body_P_sensor)) n_pass++;
    }

    std::cout << "\n========================================\n";
    std::cout << "Passed " << n_pass << " / " << n_total << " cases.\n";
    return (n_pass == n_total) ? 0 : 1;
}