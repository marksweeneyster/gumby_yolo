#pragma once

#include <cstdint>
#include <memory>

namespace cvt {

    enum PwmCmd : uint8_t {
        SET_LEVEL = 0x01,
        SET_TRIM = 0x02,
        CMD_INVALID = 0x05
    };

    enum PwmId : uint8_t {
        STEERING = 0x01,
        THROTTLE = 0x02,
        Y_AXIS = 0x03,
        X_AXIS_A = 0x04,
        X_AXIS_B = 0x05,
        TRIGGER = 0x06,
        FLYWHEEL = 0x07,
        ID_INVALID = 0x08
    };

    class I2cBus {
    public:
        static std::shared_ptr<I2cBus> Instance();

        int writeData(uint8_t* buf, size_t len);

        int16_t horizontalLevel;
        int16_t verticalLevel;

        static std::shared_ptr<I2cBus> m_bus;
    private:
        I2cBus() { } 

        void i2cStart(const std::string& filename, uint8_t slaveAddress);

        int m_i2cFd;
        uint8_t m_slaveAddress;
    };
}
