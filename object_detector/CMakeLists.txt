cmake_minimum_required(VERSION 3.16)
project(ocv_rs2_obj_detector)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

find_package(realsense2)

add_subdirectory(i2c)
add_subdirectory(cvt_hal)

if(realsense2_FOUND)
    message(STATUS "RS2: ${realsense2_LIBRARY}")
    message(STATUS "RS2: ${realsense2_INCLUDE_DIR}")

    add_executable(ocv_rs2_obj_detector main.cpp)

    target_include_directories(ocv_rs2_obj_detector PUBLIC
            ocv_utilities
    )
    target_link_libraries(ocv_rs2_obj_detector PUBLIC
            ${realsense2_LIBRARY}
            ${OpenCV_LIBS}
    )

    add_executable(cvt cvt.cpp)
    target_include_directories(cvt PUBLIC
            ocv_utilities
            cvt_hal
    )
    target_link_libraries(cvt PUBLIC
            ${realsense2_LIBRARY}
            ${OpenCV_LIBS}
            cvt_hal
    )
else()
    message(STATUS "Not building the RealSense app")
endif()

#add_executable(ocv_obj_detector webcam_main.cpp)
#target_include_directories(ocv_obj_detector PUBLIC
#        ocv_utilities
#)
#target_link_libraries(ocv_obj_detector PUBLIC
#        ${OpenCV_LIBS}
#)
#
#add_executable(ocv_fr_obj_detector file_reader_main.cpp)
#target_include_directories(ocv_fr_obj_detector PUBLIC
#        ocv_utilities
#)
#target_link_libraries(ocv_fr_obj_detector PUBLIC
#        ${OpenCV_LIBS}
#)
#
