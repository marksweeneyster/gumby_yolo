
#ifndef OCV_RS2_OBJ_DETECTOR_FOUNDOBJECT_HPP
#define OCV_RS2_OBJ_DETECTOR_FOUNDOBJECT_HPP

#include <vector>
#include <opencv2/core.hpp>
#include <librealsense2/rs.hpp>

namespace cvt {

    struct FoundObject {
        cv::Rect fbx;
        float    fconf;
        int      fid;
        float    fdist;
        constexpr static float BAD_DISTANCE = 9999.9f;

        // realsense color camera spec for field of view angles in degrees
        float horzHalfFOV = 39.7f;    // SDK for M.S. camera 70.5774612, 43.4392319 => 35.29, 21.72
        float vertHalfFOV = 21.25f;
        constexpr static float BAD_ANGLE = -999.9f;

        explicit FoundObject(cv::Rect bx, float conf, int id, float dist) {
            fbx   = bx;
            fconf = conf;
            fid   = id;
            fdist = dist > 0.0f ? dist : BAD_DISTANCE;
        }
        explicit FoundObject() {
            fconf = 0.0f;
            fid   = -1;
            fdist = BAD_DISTANCE;
        }
        /**
         * Set the field of view angles in degrees (half-range)
         * @param horizontal
         * @param vertical
         */
        void setFOV(float horizontal, float vertical) {
            horzHalfFOV = horizontal;
            vertHalfFOV = vertical;
        }

        [[nodiscard]]
        float getHorizAngle(int imgCols) const {
            if (imgCols < 2) {
                return BAD_ANGLE;
            }
            auto cx = static_cast<float>(imgCols)*0.5f;

            auto xpix = static_cast<float>(fbx.x) + 0.5f*static_cast<float>(fbx.width) - cx;
            auto x = xpix/cx;

            return x*horzHalfFOV;
        }

        float getVertAngle(int imgRows) const {
            if (imgRows < 2) {
                return BAD_ANGLE;
            }
            auto cy = static_cast<float>(imgRows) * 0.5f;

            auto ypix = static_cast<float>(fbx.y) + 0.5f * static_cast<float>(fbx.height) - cy;
            auto y = -1.f*ypix/cy; // make "up" the positive y-direction

            return y*vertHalfFOV;
        }

        std::pair<int,int> getCenter() {
            auto x = fbx.x + static_cast<int>(static_cast<float>(fbx.width)*0.5f + 0.5f);
            auto y = fbx.y + static_cast<int>(static_cast<float>(fbx.height)*0.5f + 0.5f);
            return std::make_pair(x,y);
        }

        bool overlap(const FoundObject& B) const {
            auto x  = fbx.x;
            auto y  = fbx.y;
            auto w  = fbx.width;
            auto h = fbx.height;

            auto dw  = fbx.width / 20;
            auto dh = fbx.height / 20;

            auto p1 = std::make_pair(x + dw, y + dh);
            auto p2 = std::make_pair(x + w - dw, y + dh);
            auto p3 = std::make_pair(x + w - dw, y + h - dh);
            auto p4 = std::make_pair(x + dw, y + h - dh);

            auto xB1 = B.fbx.x;
            auto yB1 = B.fbx.y;
            auto xB2 = xB1 + B.fbx.width;
            auto yB2 = yB1 + B.fbx.height;

            if ( p1.first > xB1 && p1.first < xB2 && p1.second > yB1 && p1.second < yB2) {
                return true;
            } else if ( p2.first > xB1 && p2.first < xB2 && p2.second > yB1 && p2.second < yB2) {
                return true;
            } else if ( p3.first > xB1 && p3.first < xB2 && p3.second > yB1 && p3.second < yB2) {
                return true;
            } else if ( p4.first > xB1 && p4.first < xB2 && p4.second > yB1 && p4.second < yB2) {
                return true;
            }

            return false;
        }
    };

    /**
     * Return the horizontal and vertical angles for the object with the highest confidence that matches index.
     * If there are no matches then return {FoundObject::BAD_ANGLE, FoundObject::BAD_ANGLE}.
     * FOV half angles for a particular device can be obtained via the realsense call: rs2_fov.  If these parameters
     * aren't entered then object angle calculation uses the datasheet values for FOV.
     *
     * @param objects  vector of FoundObjects returned by cvt::postprocess
     * @param index    the class index
     * @param imgRows  rows in the image where objects were found
     * @param imgCols  columns in the image where objects were found
     * @param horzHalfFOV  horizontal FOV half-angle in degrees
     * @param vertHalfFOV  vertical FOV half-angle in degrees
     * @return         a pair of floats with the horizontal and vertical angles in degrees
     */
    std::pair<float,float> getTargetAngles(const std::vector<FoundObject>& objects,
                                           int index, int imgRows, int imgCols,
                                           float horzHalfFOV = -1.0f, float vertHalfFOV = -1.0f) {

        std::vector<FoundObject> matches;
        auto pred = [index](const FoundObject& obj){ return obj.fid == index; };
        std::copy_if(objects.begin(), objects.end(), std::back_inserter(matches), pred);

        FoundObject target;
        for (auto& match : matches) {
            if (match.fconf > target.fconf) {
                target = match;
            }
        }
        float horizAngle = FoundObject::BAD_ANGLE;
        float vertAngle = FoundObject::BAD_ANGLE;

        if (target.fconf > 0.0f) {
            if (horzHalfFOV>0.0f && vertHalfFOV>0.0f) {
                target.setFOV(horzHalfFOV, vertHalfFOV);
            }
            horizAngle = target.getHorizAngle(imgCols);
            vertAngle = target.getVertAngle(imgRows);
        }
        return std::make_pair(horizAngle, vertAngle);
    }
}

#endif //OCV_RS2_OBJ_DETECTOR_FOUNDOBJECT_HPP
