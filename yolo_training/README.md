# YOLOv3 Training

- **gumby_pokey_training.ipynb**: notebook for training the two class model on [Google Colab](https://colab.research.google.com/)
- **gumby_n_friends_training.ipynb**: notebook for training the five class model
- **yolov3_TWO_classes.cfg**: config file for running object detection for 2 classes
- **yolov3_FIVE_classes.cfg**: config file for running object detection for 5 classes

We can use a config file that has fewer classes than the weights file.  Using **yolov3_TWO_classes.cfg** with a 
weights file trained for five classes means that the object detector will never "find" the last three classes. This 
can be useful if we are trying to eliminate false positives that might show up for a weights files that was trained just
on those two classes.    

YOLOv3 identifies a class by an integer id.  Models used in this project have the following class ids: 

0. Gumby
1. Pokey
2. Minga
3. Goo
4. Prickle

~~Weight files can be found [here](https://drive.google.com/drive/folders/1Z4tikenkAGshfk9R0B77TEgwZcyqGEE8?usp=sharing).~~
