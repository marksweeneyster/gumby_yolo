#ifndef OCV_UTILS_HPP
#define OCV_UTILS_HPP

#include <exception>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/opencv.hpp>   // Include OpenCV API

// Convert rs2::frame to cv::Mat
static cv::Mat frame_to_mat(const rs2::frame &f) {
    using namespace cv;
    using namespace rs2;

    auto vf = f.as<video_frame>();
    const int w = vf.get_width();
    const int h = vf.get_height();

    if (f.get_profile().format() == RS2_FORMAT_BGR8) {
        return {Size(w, h), CV_8UC3, (void *)f.get_data(), Mat::AUTO_STEP};
    } else if (f.get_profile().format() == RS2_FORMAT_RGB8) {
        auto r_rgb = Mat(Size(w, h), CV_8UC3, (void *)f.get_data(), Mat::AUTO_STEP);
        Mat r_bgr;
        cvtColor(r_rgb, r_bgr, COLOR_RGB2BGR);
        return r_bgr;
    } else if (f.get_profile().format() == RS2_FORMAT_Z16) {
        return {Size(w, h), CV_16UC1, (void *)f.get_data(), Mat::AUTO_STEP};
    } else if (f.get_profile().format() == RS2_FORMAT_Y8) {
        return {Size(w, h), CV_8UC1, (void *)f.get_data(), Mat::AUTO_STEP};
    } else if (f.get_profile().format() == RS2_FORMAT_DISPARITY32) {
        return {Size(w, h), CV_32FC1, (void *)f.get_data(), Mat::AUTO_STEP};
    } else if (f.get_profile().format() == RS2_FORMAT_XYZ32F) {
        return {Size(w, h), CV_32FC3, (void *)f.get_data(), Mat::AUTO_STEP};
    }

    throw std::runtime_error("Frame format is not supported yet!");
}

static cv::Mat depth_frame_to_xyz(const rs2::depth_frame& depth) {
    using namespace cv;
    using namespace rs2;

    const int w = depth.get_width();
    const int h = depth.get_height();
    const int stride = depth.get_stride_in_bytes();

    rs2::pointcloud pc;
    rs2::points points = pc.calculate(depth);

    auto vertices = points.get_vertices();

    int leftmid = w * h / 2;
    int mid = w * h / 2 + w / 2;
    int rightmid = w * h / 2 + w - 1;
    auto vlmid = vertices[leftmid];
    auto vmid = vertices[mid];
    auto vrmid = vertices[rightmid];

    return {Size(w, h), CV_32FC3, (void *)points.get_data(), Mat::AUTO_STEP};
}

// Converts depth frame to a matrix of doubles with distances in meters
static cv::Mat depth_frame_to_meters(const rs2::depth_frame &f) {
    cv::Mat dm = frame_to_mat(f);
    dm.convertTo(dm, CV_64F);
    dm = dm * f.get_units();
    return dm;
}

namespace cvt {
    static float get_depth_scale(const rs2::device& dev) {
        // Go over the device's sensors
        for (rs2::sensor &sensor : dev.query_sensors()) {
            // Check if the sensor is a depth sensor
            if (auto dpt = sensor.as<rs2::depth_sensor>()) {
                return dpt.get_depth_scale();
            }
        }
        throw std::runtime_error("Device does not have a depth sensor");
    }
} // namespace cvt

#endif // OCV_UTILS_HPP
