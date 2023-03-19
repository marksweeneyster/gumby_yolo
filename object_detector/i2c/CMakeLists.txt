set(THIS_TARGET "not_i2c")


#target_include_directories(${THIS_TARGET} ${CMAKE_CURRENT_LISTS_DIR})

add_library(${THIS_TARGET} i2c.cpp)

target_link_libraries(${THIS_TARGET} PUBLIC i2c)

