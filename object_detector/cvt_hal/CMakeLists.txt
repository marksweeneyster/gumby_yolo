set(THIS_TARGET "cvt_hal")



add_library(${THIS_TARGET} hal.cpp)

target_link_libraries(${THIS_TARGET} PUBLIC not_i2c)
