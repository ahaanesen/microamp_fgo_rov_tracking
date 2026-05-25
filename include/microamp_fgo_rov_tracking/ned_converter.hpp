#pragma once
#include <cmath>

struct NedCoordinates {
    double n; // North
    double e; // East
    double d; // Down
};

struct NedConverter
{
    // Datum (lat0, lon0 in rad, h0 in meters)
    double lat0_rad, lon0_rad;
    double sin_lat0, cos_lat0;
    double sin_lon0, cos_lon0;

    bool initialised = false;

    // WGS84 constants
    static constexpr double r_e = 6378137.0;
    static constexpr double f = 1.0 / 298.257223563;
    static constexpr double e2 = f * (2 - f);

    void setDatum(double lat_deg, double lon_deg, double h_m)
    {
        lat0_rad = lat_deg * M_PI / 180.0;
        lon0_rad = lon_deg * M_PI / 180.0;

        sin_lat0 = std::sin(lat0_rad);
        cos_lat0 = std::cos(lat0_rad);
        sin_lon0 = std::sin(lon0_rad);
        cos_lon0 = std::cos(lon0_rad);

        // Precompute datum ECEF
        double N0 = r_e / std::sqrt(1.0 - e2 * sin_lat0 * sin_lat0);
        x0 = (N0 + h_m) * cos_lat0 * cos_lon0;
        y0 = (N0 + h_m) * cos_lat0 * sin_lon0;
        z0 = (N0 * (1 - e2) + h_m) * sin_lat0;

        initialised = true;
    }

    // Convert a GNSS fix to local NED coordinates
    NedCoordinates gnssToNED(double lat_deg, double lon_deg, double h_m)
    {
        double lat = lat_deg * M_PI / 180.0;
        double lon = lon_deg * M_PI / 180.0;

        double sin_lat = std::sin(lat);
        double cos_lat = std::cos(lat);
        double sin_lon = std::sin(lon);
        double cos_lon = std::cos(lon);

        double N = r_e / std::sqrt(1 - e2 * sin_lat * sin_lat);

        double x = (N + h_m) * cos_lat * cos_lon;
        double y = (N + h_m) * cos_lat * sin_lon;
        double z = (N * (1 - e2) + h_m) * sin_lat;

        // delta ECEF
        double dx = x - x0;
        double dy = y - y0;
        double dz = z - z0;

        double n, e, d;
        // Rotate ECEF → NED
        n =  -sin_lat0 * cos_lon0 * dx
             -sin_lat0 * sin_lon0 * dy
             +cos_lat0 * dz;

        e =  -sin_lon0 * dx
             +cos_lon0 * dy;

        d =  -cos_lat0 * cos_lon0 * dx
             -cos_lat0 * sin_lon0 * dy
             -sin_lat0 * dz;
        return NedCoordinates{n, e, d};
    }

private:
    double x0, y0, z0;
};