
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "ocv_utils.hpp"
#include "cvt_utils.hpp"
#include "hal.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

namespace cvt {

    std::pair<rs2::vertex,FoundObject> getTargetXYZ(const std::vector<FoundObject>& objects, int index, const rs2::depth_frame& depth) {
        std::vector<FoundObject> matches;
        auto pred = [index](const FoundObject& obj){ return obj.fid == index; };
        std::copy_if(objects.begin(), objects.end(), std::back_inserter(matches), pred);

        std::vector<FoundObject> nonMatches;
        auto pred2 = [index](const FoundObject& obj){ return obj.fid != index; };
        std::copy_if(objects.begin(), objects.end(), std::back_inserter(nonMatches), pred2);

        FoundObject target;
        for (auto& match : matches) {
            bool overlap = false;
            for(auto& nonMatch : nonMatches) {
                if ( match.overlap(nonMatch)) {
                    overlap = true;
                    std::cout << "getTargetXYZ OVERLAP\n";
                }
            }
            if (!overlap && match.fconf > target.fconf) {
                target = match;
            }
        }
        if (target.fid == -1) {
            return std::make_pair(rs2::vertex{0.f,0.f,0.f}, target);
        }

        auto targetCenter = target.getCenter();

        auto w = depth.get_width();

        rs2::pointcloud pc;
        rs2::points points = pc.calculate(depth);

        auto vertices = points.get_vertices();

        auto offset = w*targetCenter.second + targetCenter.first;

        return std::make_pair(vertices[offset], target);
    }

}

const char* keys =
        "{help h usage ? | | Usage example: \n\t\t. --config=yolov3.cfg --weights=yolov3.weights --backend=gpu --model=gumby}"
        "{config c        |yolov3.cfg| input YOLO model config file   }"
        "{weights w       |yolov3.weights| input YOLO model weights file   }"
        "{backend b       |gpu| input backend (cpu or gpu)   }"
        "{model m         |gumby| input YOLO model type (gumby or coco)   }";

int main(int argc, char * argv[]) try
{
    auto cvtState = cvt::State::stopped;
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Application to detect objects with a YOLO model and the Intel RealSense2 camera.");
    if (parser.has("help") || argc == 1) {
        parser.printMessage();
        return 0;
    }
    auto modelConfigFile = parser.get<cv::String>("config");
    auto modelWeightsFile = parser.get<cv::String>("weights");
    auto net_backend = parser.get<cv::String>("backend");
    auto modelTypeName = parser.get<cv::String>("model");

    // TODO below hardcoded values are preliminary, re-check before show time
    /**
     * right handed coordinate system
     * Looking out from the camera is +z
     * Right is +x, Left is -x
     * Up -y, down is +y
     *
     * Offsets below are vectors from the rotation axis to the camera in meters
     */
    // Azimuth rotates in the horizontal plane (we walk around in the horizontal plane)
    // Post tape-job#1
    float AzimuthCameraOffset[] {-0.02f,-0.252f,0.195f};
    // Elevation rotates up/down relative to the horizontal plane
    float ElevationCameraOffset[] {0.0f,-0.0f,0.14f};

    cvt::Model modelType = cvt::Model::gumby;

    if (modelTypeName.find("coco") != std::string::npos) {
        modelType = cvt::Model::coco;
    }

    cvt::setModelType(modelType);

    // Load the network
    auto net = cv::dnn::readNetFromDarknet(modelConfigFile, modelWeightsFile);
    if (net_backend == "gpu") {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    /////////////////////////////

    rs2::log_to_console(RS2_LOG_SEVERITY_ERROR);

    rs2::config rs2_cfg;

    // TODO it's an open question on whether changing the default 1280x720 for rs2 improves object detection.
    // sizes common to depth and rgb
    // 1280x720
    // 848x480
    // 640x480
    // 640x360
    // 424x240
    int requestedWidth     = 1280;
    int requestedHeight    = 720;
    int requestedFrameRate = 30;

    rs2_cfg.enable_stream(RS2_STREAM_COLOR, requestedWidth, requestedHeight, RS2_FORMAT_BGR8, requestedFrameRate);
    // TODO if we're not using depth then may don't enable
    rs2_cfg.enable_stream(RS2_STREAM_DEPTH, requestedWidth, requestedHeight, RS2_FORMAT_Z16, requestedFrameRate);

    rs2::pipeline pipe;

    rs2::pipeline_profile profile = pipe.start(rs2_cfg);

    auto depth_stream = profile.get_stream(RS2_STREAM_COLOR)
            .as<rs2::video_stream_profile>();
    auto intrinsics = depth_stream.get_intrinsics();
    float fov[2]; // X, Y fov
    rs2_fov(&intrinsics, fov);
    const float horzHalfFOV = 0.5f*fov[0];
    const float vertHalfFOV = 0.5f*fov[1];

    std::cout << "angles: [" << fov[0] << ", " << fov[1] << "]\n";

    // Block program until frames arrive
    auto frames = pipe.wait_for_frames();

    auto rgb = frames.get_color_frame();

    // Get the depth frame's dimensions
    auto width  = rgb.get_width();
    auto height = rgb.get_height();

    std::cout << "rows: " << height << ", cols: " << width << '\n';

    float depth_scale = cvt::get_depth_scale(profile.get_device());
    std::cout << "depth_scale: " << depth_scale << '\n';

    rs2::align align_to_color(RS2_STREAM_COLOR);

    // Create a window
    static const std::string kWinName = "CVT RealSense";
    namedWindow(kWinName, cv::WINDOW_FULLSCREEN);

    cv::Mat color_mat, depth_mat, blob;
    cvt::FoundObject lastTarget;

    const int GumbyIndex = 0;
    int imageIndx = 0;


    cvt::RandomWalk(true);
    cvtState = cvt::State::scanning;
    while (getWindowProperty(kWinName, cv::WND_PROP_AUTOSIZE) >= 0)
    {
        frames = pipe.wait_for_frames();
        frames = align_to_color.process(frames);

        rgb = frames.get_color_frame();
        color_mat = frame_to_mat(rgb);

        auto depth = frames.get_depth_frame();
        depth_mat = frame_to_mat(depth);

        cv::dnn::blobFromImage(color_mat, blob, cvt::BLOB_SCALE, cv::Size(cvt::NET_IMAGE_WIDTH, cvt::NET_IMAGE_HEIGHT), cv::Scalar(0,0,0), true, false);

        //Sets the input to the network
        net.setInput(blob);

        // Runs the forward pass to get output of the output layers
        std::vector<cv::Mat> outs;
        net.forward(outs, cvt::getOutputsNames(net));

        // Remove the bounding boxes with low confidence
        auto objects = cvt::postprocess(color_mat, outs, true, depth_mat, depth_scale);

        if (cvtState == cvt::State::aligning) {
            // skip a frame for target testing if we are aligning
            cvtState = cvt::State::scanning;
        } else if (objects.empty()) {
            if (cvtState == cvt::State::firing) {
                std::cout << "check\n";
            }
            cvtState = cvt::State::scanning;
        } else {
            auto xyzTarget = cvt::getTargetXYZ(objects, GumbyIndex, depth);
            auto vertex = xyzTarget.first;
            auto target = xyzTarget.second;

            if (target.fid == GumbyIndex) {
                if (cvtState == cvt::State::firing && !target.overlap(lastTarget)) {
                    std::cout << "check\n";
                }
                lastTarget = target;

                std::cout << "Gumby position(m): (" << vertex.x << "," << vertex.y << "," << vertex.z << ")\n";

                float fy = vertex.y + ElevationCameraOffset[1];
                float fz = vertex.z + ElevationCameraOffset[2];

                float elevation = std::atan2f(fy, fz) * cvt::RAD_TO_DEGREE;
                float fx = vertex.x + AzimuthCameraOffset[0];

                fz = vertex.z + AzimuthCameraOffset[2];

                float azimuth = std::atan2f(fx, fz) * cvt::RAD_TO_DEGREE;

                std::cout << "\nTurret angles(azim,elev): [" << azimuth << ", " << elevation << "] \n\n";

                auto fabsElev = fabs(elevation);
                auto fabsAzim = fabs(azimuth);
                if (fabsElev < cvt::CONE_OF_DEATH_DEGREES && fabsAzim < cvt::CONE_OF_DEATH_DEGREES) {
                    cvtState = cvt::State::firing;
                    cvt::Fire();
                    putText(color_mat, "Fire!", cv::Point(width * 9 / 20, height / 2), cv::FONT_HERSHEY_SIMPLEX, 2.0,
                            cv::Scalar(0, 0, 255));
                } else if (fabsAzim > 2.0f) {
                    cvtState = cvt::State::aligning;
                    cvt::RotateTurretHorizontal(-1.0f * azimuth);
                } else {
                    cvtState = cvt::State::aligning;
                    cvt::RotateTurretVertical(elevation);
                }
            } else if (cvtState == cvt::State::firing) {
                std::cout << "check\n";
                cvtState = cvt::State::scanning;
            }
        }

        // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        std::vector<double> layersTimes;
        double freq = cv::getTickFrequency() * 0.001;
        double t = static_cast<double>(net.getPerfProfile(layersTimes)) / freq;
        std::string label = cv::format("Inference time for a frame : %.2f ms", t);
        putText(color_mat, label, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));

        imshow(kWinName, color_mat);
        if (cv::waitKey(1) >= 0) break;
    }

    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
