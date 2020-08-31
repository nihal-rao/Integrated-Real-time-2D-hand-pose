# An Integrated Method for Realtime 2D Hand Pose Detection

<p align="left">
  <img width="500" height="500" src="https://github.com/nihal-rao/Integrated-Real-time-2D-hand-pose/blob/master/images/LSMV.png">
</p>

<p align="left">
  <img width="500" height="500" src="https://github.com/nihal-rao/Integrated-Real-time-2D-hand-pose/blob/master/images/NZSL_1.png">
</p>

We present an integrated, real-time approach for 2D hand pose detection from a monocular RGB image, with a common backbone shared between the bounding box detector and the keypoint detector subnets. This is in contrast to traditional methods which use two separate models for hand localization and keypoint detection with no sharing of features. We build on the popular RetinaNet architecture for object detection and introduce an integrated model which performs both hand localization and keypoint detection in real-time. We evaluate our approach on two different datasets and show evidence that our model obtains accurate results.

The files defining the new architecture can be found in the folder KP_RN_Configs folder.

Models for LSMV and NZSL datasets can be found [here](https://drive.google.com/drive/folders/1TFRtcexID1f9uo-bZC-JqWxMP4XqKdGH?usp=sharing).

## Requirements
* Python 3.6+
* We use detectron2 v0.1.1 for all experiments. 
* For usage on local machines , clone this repository and install the packages in requirements.txt
* For ease of usage on colab notebooks, add the KP_RN_Configs folder to your Google drive and follow the training notebooks for setup.

## Datasets
* To use your own dataset, convert keypoint and hand bbox annotations to COCO format.
* We use the [LSMV](http://www.rovit.ua.es/dataset/mhpdataset/) dataset and the [NZSL](http://domedb.perception.cs.cmu.edu/handdb.html) dataset.

## Usage 
For a detailed guide on training and evaluation please go through the colab notebooks added above.


