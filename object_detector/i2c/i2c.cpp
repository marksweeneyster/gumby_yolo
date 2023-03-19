#include "i2c.hpp"

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

std::shared_ptr<I2cBus> I2cBus::m_bus = {};

std::shared_ptr<I2cBus> I2cBus::Instance() {
    if (!m_bus) {
        m_bus = std::shared_ptr<I2cBus>(new I2cBus);
        m_bus->m_i2cFd = 0;
        m_bus->m_slaveAddress = 0x17;
        m_bus->horizontalLevel = 1500;
        m_bus->verticalLevel = 1500;
        m_bus->i2cStart("/dev/i2c-8", m_bus->m_slaveAddress);
    }
    return m_bus;
}

int I2cBus::writeData(uint8_t* buf, size_t len) {
    if (m_i2cFd <= 0) {
         std::cerr << "WriteData will fail" << std::endl;
    }
   
    return write(m_i2cFd, buf, len);
}


void I2cBus::i2cStart(const std::string& filename, uint8_t slaveAddress) {
    int rc;
    m_i2cFd = open(filename.c_str(), O_RDWR);
    if (m_i2cFd < 0) {
        rc = m_i2cFd;
        std::cerr<<"failed to open i2c"<<std::endl;
    }
}

} // End namespace cvt
