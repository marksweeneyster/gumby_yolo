# OpenCV Deep Neural Net Object Detector with Intel RealSense2

## To build

You'll need opencv and realsense2 installed and pont cmake to dirs holding their respective *Config.cmake files:
```asm
mkdir .build && cd .build
cmake .. -DOpenCV_DIR=<path to OpenCVConfig.cmake> -Drealsense2_DIR=<path to realsense2Config.cmake>
```
We need the OpenCV DNN module.

## To run 

You'll need a YOLO model weight file and a YOLO model config file (cfg).
You can get them [here](https://pjreddie.com/darknet/yolo/), get the `YOLOv3-416.*` files. The object detector network is currently hardcoded for the 416 size.

Gumby and fiends models can be found [here](https://drive.google.com/drive/folders/1Z4tikenkAGshfk9R0B77TEgwZcyqGEE8?usp=sharing).

### Intel RealSense camera
```asm
 ./ocv_rs2_obj_detector --config=/home/mark/models/yolov3.cfg \
 --weights=/home/mark/Dev/models/yolov3.weights \
 --backend=gpu --model=gumby
```

The `backend` setting will fall back to cpu if the OpenCV DNN is not built for GPU.
Currently, any gpu type is supported as long as it's NVidia (Cuda).  
The `model` setting chooses the object model **gumby** or **coco** (80 class model). The model needs to match the config 
and weights file.


### Generic cameras
There is also a webcam only version that can grab images from most laptop or desktop cameras.

```asm
 ./ocv_obj_detector --config=/home/mark/models/yolov3.cfg \
 --weights=/home/mark/Dev/models/yolov3.weights \
 --backend=gpu --model=gumby --camera=0
```
If you have multiple camera then you will need to experiment with the `camera` setting 
to find which index goes with which camera.

### File reader
There is also an app that will iterate over all image files in a directory
and run the object detection.
```asm
./cv_fr_obj_detector.exe  --config=yolov3_FIVE_classes.cfg \
 --weights=yolov3_FIVE_classes.weights \
 --directory=/home/mark/data/gumby_n_friends  --backend=cpu
```
Type `q` to exit the app, hit any other key to step forward through the images.
