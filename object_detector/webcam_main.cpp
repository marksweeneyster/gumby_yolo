//This file is part of gumby_yolo, an object detector that works with various cameras.
// Copyright (C) 2024 Mark Sweeney, marksweeneyster@gmail.com
//
// gumby_yolo is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.


#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "cvt_utils.hpp"
#include <iostream>
#include <vector>

const char* keys =
        "{help h usage ? | | Usage example: \n\t\t. --config=yolov3.cfg --weights=yolov3.weights --backend=gpu --camera=0 --model=gumby}"
        "{config c        |yolov3.cfg| input YOLO model config file   }"
        "{weights w       |yolov3.weights| input YOLO model weights file   }"
        "{backend b       |gpu| input backend (cpu or gpu)   }"
        "{model m         |gumby| input YOLO model type (gumby or coco)   }"
        "{camera w        |0| input Camera index }";


int main(int argc, char * argv[]) try {

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Application to detect objects with a YOLO model and images from a camera.");
    if (parser.has("help") || argc == 1) {
        parser.printMessage();
        return 0;
    }
    auto modelConfigFile = parser.get<cv::String>("config");
    auto modelWeightsFile = parser.get<cv::String>("weights");
    auto net_backend = parser.get<cv::String>("backend");
    auto modelTypeName = parser.get<cv::String>("model");
    auto cameraIndex = parser.get<int>("camera");

    std::cout << "camera index " << cameraIndex << '\n';

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

    // Create a window
    static const std::string kWinName = "CVT WebCam";
    namedWindow(kWinName, cv::WINDOW_FULLSCREEN);

    cv::Mat color_mat, blob;

    cv::VideoCapture cap(cameraIndex);

    if (!cap.isOpened()) {
        std:: cout << "cannot open camera";
        return 0;
    }

    while (getWindowProperty(kWinName, cv::WND_PROP_AUTOSIZE) >= 0) {
        cap >> color_mat;

        cv::dnn::blobFromImage(color_mat, blob, cvt::BLOB_SCALE,
                               cv::Size(cvt::NET_IMAGE_WIDTH, cvt::NET_IMAGE_HEIGHT),
                               cv::Scalar(0,0,0), true, false);
        //Sets the input to the network
        net.setInput(blob);
        // Runs the forward pass to get output of the output layers
        std::vector<cv::Mat> outs;
        net.forward(outs, cvt::getOutputsNames(net));

        // Remove the bounding boxes with low confidence
        auto objects = cvt::postprocess(color_mat, outs, true);
        // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        std::vector<double> layersTimes;
        double freq = cv::getTickFrequency() * 0.001;
        double t = static_cast<double>(net.getPerfProfile(layersTimes)) / freq;
        std::string label = cv::format("Inference time for a frame : %.2f ms", t);
        putText(color_mat, label, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));

        imshow(kWinName, color_mat);
        if (cv::waitKey(1) >= 0) break;
    }


    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}