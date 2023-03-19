
#ifndef OCV_RS2_OBJ_DETECTOR_HAL_HPP
#define OCV_RS2_OBJ_DETECTOR_HAL_HPP

namespace cvt {
    /**
     * Rotate the turret in the horizontal plane. Positive values are to the right, negative values are to the left.
     * @param degrees
     */
    void RotateTurretHorizontal(float degrees);

    /**
     * Rotate the turret in the vertical plane
    * @param degrees
    */
    void RotateTurretVertical(float degrees);

    /**
     * Reset turret to neutral position
     */
    void Reset();

    void RandomWalk(bool start);

    void Fire();
}
#endif //OCV_RS2_OBJ_DETECTOR_HAL_HPP
