#include "hal.hpp"
#include "../i2c/i2c.hpp"

#include <sys/ioctl.h>
#include <linux/i2c.h>
#include <linux/i2c-dev.h>

#include <iostream>
#include <fstream>
#include <memory>

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

namespace cvt {

    void RotateTurretHorizontal(float degrees) {
        std::cout << "cvt::RotateTurretHorizontal " << degrees << " degrees\n";

        // TODO: scale degrees into pwm range
        // -90f <==> +90f
        // 1000 <==> 2000
        constexpr float halfLevelRange = (2000 - 1000) / 2;
        constexpr float degreeMaxRange = 90.0;
        constexpr float levelsPerDegree = halfLevelRange / degreeMaxRange; 

        // Each new call is relative so add to our stored value
        int16_t level = degrees * levelsPerDegree;

        std::cout << "Horizontal level: " << level << std::endl;
        auto i2c = I2cBus::Instance();
        i2c->horizontalLevel += level;
        std::cout << "Accumulated horizontal level: " << i2c->horizontalLevel << std::endl;
        uint8_t buf[4] {
            PwmId::Y_AXIS,
            PwmCmd::SET_LEVEL,
            static_cast<uint8_t>(i2c->horizontalLevel >> 8),
            static_cast<uint8_t>(i2c->horizontalLevel & 0x00FF)
        };
        i2c->writeData(buf, 4);
    }

    void RotateTurretVertical(float degrees) {
        std::cout << "cvt::RotateTurretVertical " << degrees << " degrees\n";
    }

    void Reset() {
        std::cout << "cvt::Reset\n";
    }

    void RandomWalk(bool start) {
        std::cout << "cvt::RandomWalk start? " << start << "\n";
        std::cerr << "cvt::RandomWalk nope ESC is broken" << std::endl;
    }

    void Fire() {
        std::cout << "cvt::Fire\n";
    }
}
