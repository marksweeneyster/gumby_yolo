#ifndef CVT_UTILS_HPP
#define CVT_UTILS_HPP

#include "FoundObject.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <string>
#include <vector>

namespace cvt {
// Initialize the parameters
    static constexpr float CONF_THRESHOLD   = 0.5f; // Confidence threshold
    static constexpr float NMS_THRESHOLD    = 0.4f; // Non-maximum suppression threshold
    static constexpr int   NET_IMAGE_WIDTH  = 416;  // Width of network's input image
    static constexpr int   NET_IMAGE_HEIGHT = 416;  // Height of network's input image

    static constexpr double BLOB_SCALE = 1.0 / 255.0;

    static constexpr float RAD_TO_DEGREE = 57.2957796f;
    static constexpr float CONE_OF_DEATH_DEGREES = 1.0f; // this is a radius, cone width is 2x

    enum class Model { gumby, coco };
    enum class State { stopped, scanning, aligning, firing };

    const std::vector<std::string> COCO_CLASS_LABELS{
            "person",       "bicycle",      "car",          "motorcycle",
            "airplane",     "bus",          "train",        "truck",
            "boat",         "trafficlight", "firehydrant",  "stopsign",
            "parkingmeter", "bench",        "bird",         "cat",
            "dog",          "horse",        "sheep",        "cow",
            "elephant",     "bear",         "zebra",        "giraffe",
            "backpack",     "umbrella",     "handbag",      "tie",
            "suitcase",     "frisbee",      "skis",         "snowboard",
            "sportsball",   "kite",         "baseballbat",  "baseballglove",
            "skateboard",   "surfboard",    "tennisracket", "bottle",
            "wineglass",    "cup",          "fork",         "knife",
            "spoon",        "bowl",         "banana",       "apple",
            "sandwich",     "orange",       "broccoli",     "carrot",
            "hotdog",       "pizza",        "donut",        "cake",
            "chair",        "sofa",         "pottedplant",  "bed",
            "diningtable",  "toilet",       "tvmonitor",    "laptop",
            "mouse",        "remote",       "keyboard",     "cellphone",
            "microwave",    "oven",         "toaster",      "sink",
            "refrigerator", "book",         "clock",        "vase",
            "scissors",     "teddybear",    "hairdrier",    "toothbrush"};

// The last 3 classes are intended to prevent Gumby and Pokey false positives.
// We may want to remove them from this list, so we only report a Gumbey or a
// Pokey.
    static const std::vector<std::string> GUMBY_CLASS_LABELS{
            "Gumby", "Pokey", "Minga", "Goo", "Prickle"};

    const std::vector<cv::Scalar> COLORS{
            cv::Scalar(0, 255, 0),     cv::Scalar(0, 69, 255),
            cv::Scalar(127, 127, 255), cv::Scalar(255, 0, 0),
            cv::Scalar(0, 255, 255),   cv::Scalar(0, 0, 255),
            cv::Scalar(255, 255, 0),   cv::Scalar(255, 200, 20),
            cv::Scalar(255, 0, 255),   cv::Scalar(127, 0, 255),
            cv::Scalar(0, 127, 255),   cv::Scalar(20, 20, 20),
            cv::Scalar(127, 255, 0),   cv::Scalar(255, 0, 127)};

    static cv::Scalar getColor(int index) { return COLORS[index % COLORS.size()]; }

    std::vector<std::string> classes = cvt::GUMBY_CLASS_LABELS;

    static void setModelType(Model type) {
        if (type == Model::gumby) {
            classes = cvt::GUMBY_CLASS_LABELS;
        } else {
            classes = cvt::COCO_CLASS_LABELS;
        }
    }

// Draw the predicted bounding box
    static void drawPred(int classId, float conf, int left, int top, int right,
                         int bottom, const cv::Mat &frame, float distance) {
        cv::Scalar color = cvt::getColor(classId);
        // Draw a rectangle displaying the bounding box
        cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), color, 3);

        // Get the label for the class name and its confidence
        auto label = (distance > 0.0f ? cv::format("%.2f, %.2f", conf, distance) :  cv::format("%.2f", conf));
        if (!classes.empty()) {
            CV_Assert(classId < (int)classes.size());
            label = classes[classId] + ":" + label;
        }

        // Display the label at the top of the bounding box
        int baseLine;
        cv::Size labelSize =
                getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        top = cv::max(top, labelSize.height);
        int topShift = static_cast<int>(round(1.5 * labelSize.height));
        int leftShift = static_cast<int>(round(1.5 * labelSize.width));
        rectangle(frame, cv::Point(left, top - topShift),
                  cv::Point(left + leftShift, top + baseLine),
                  cv::Scalar(255, 255, 255), cv::FILLED);
        putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75,
                cv::Scalar(0, 0, 0), 1);
    }

// Remove the bounding boxes with low confidence using non-maxima suppression
    static std::vector<FoundObject> postprocess(const cv::Mat &frame,
                                                const std::vector<cv::Mat> &outs,
                                                bool drawBoxes = false,
                                                const cv::Mat &depth = cv::Mat(),
                                                float depth_scale = 0.0f) {
        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        auto fCols = static_cast<float>(frame.cols);
        auto fRows = static_cast<float>(frame.rows);

        for (const auto &out : outs) {
            // Scan through all the bounding boxes output from the network and keep only
            // the ones with high confidence scores. Assign the box's class label as the
            // class with the highest score for the box.
            auto data = (float *)out.data;
            for (int j = 0; j < out.rows; ++j, data += out.cols) {
                cv::Mat scores = out.row(j).colRange(5, out.cols);
                cv::Point classIdPoint;
                double confidence;
                // Get the value and location of the maximum score
                minMaxLoc(scores, nullptr, &confidence, nullptr, &classIdPoint);
                if (confidence > cvt::CONF_THRESHOLD) {
                    int centerX = static_cast<int>(data[0] * fCols);
                    int centerY = static_cast<int>(data[1] * fRows);
                    int width = static_cast<int>(data[2] * fCols);
                    int height = static_cast<int>(data[3] * fRows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.emplace_back(left, top, width, height);
                }
            }
        }

        // Perform non-maximum suppression to eliminate redundant overlapping boxes
        // with lower confidences
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, cvt::CONF_THRESHOLD, cvt::NMS_THRESHOLD,
                          indices);

        // found objects
        auto objs = std::vector<FoundObject>(indices.size());

        uint32_t indx = 0;
        for (int idx : indices) {
            auto box = boxes[idx];
            auto conf = confidences[idx];
            auto id = classIds[idx];

            int col = box.x + box.width / 2;
            int row = box.y + box.height / 2;

            float f = -1.0f;
            if (depth.cols == frame.cols && depth.rows == frame.rows) {
                f = depth_scale * static_cast<float>(depth.at<unsigned short>(row, col));
            }

            objs[indx].fbx = box;
            objs[indx].fconf = conf;
            objs[indx].fid = id;
            objs[indx].fdist = f;
            ++indx;

            if (drawBoxes) {
                drawPred(id, conf, box.x, box.y, box.x + box.width, box.y + box.height,
                         frame, f);
            }
        }
        return objs;
    }

// Get the names of the output layers
    static std::vector<cv::String> getOutputsNames(const cv::dnn::Net &net) {
        static std::vector<cv::String> names;
        if (names.empty()) {
            // Get the indices of the output layers, i.e. the layers with unconnected
            // outputs
            std::vector<int> outLayers = net.getUnconnectedOutLayers();

            // get the names of all the layers in the network
            std::vector<cv::String> layersNames = net.getLayerNames();

            // Get the names of the output layers in names
            names.resize(outLayers.size());
            for (size_t i = 0; i < outLayers.size(); ++i)
                names[i] = layersNames[outLayers[i] - 1];
        }
        return names;
    }
} // namespace cvt
#endif // CVT_UTILS_HPP
