# Vehicle Detection Project

This is a project for Udacity self-driving car Nanodegree program. The aim of this project is to detect the vehicles in a camera video. The implementation of the project is in the file Vehicle_Detection-YOLOv3.ipynb and main.py. Becase I have no GPU and cloud service apply fail and requirement of RAM is too big to PC, so I clip final vidio into three section. The final video output:
[First Section](https://youtu.be/ZPRDHv0rsBY), 
[Second Section](https://youtu.be/ol0e9XkYYpg), 
[Third Section](https://youtu.be/8BoAlU_MSfc)

In this README, each step in the pipeline will be explained in details.

[//]: # (Image References)
[image1]: ./output_images/YOLOv3_arch.png
[image2]: ./output_images/car_feature_map.jpg
[image3]: ./output_images/test1.jpg
[image4]: ./output_images/test2.jpg
[image5]: ./output_images/test4.jpg
[image6]: ./output_images/test4_withoutNMS.jpg
[image7]: ./output_images/BBox.png
[image8]: ./output_images/IOU.png

---
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

Because I'm using YOLOv3, so there are no HOG, but the same principle between HOG and convolution layer is that they are using gradients to find features. But convolution layer is learning how to extracted feature.

There is a example for a convolution layer after extreacted features - the feature maps.

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

Because I use YOLOv3, so I will explain the architecture as below picture.

![alt text][image1]

It use ResNet as based CNN and reach 100 layer deep, so that It can improve mAP, more detail architecture as below.

Besides, it has 3 scale output (13 * 13, 26 * 26, 52 * 52) and every scale have 3 bounding box size, smaller scale get bigger bounding box size, bigger scale get smaller bounding box size.

So smaller scale is good at searching big object, bigger scale is good at searching small object and so on...

```
layer     filters    size              input                output
    0 conv     32  3 x 3 / 1   416 x 416 x   3   ->   416 x 416 x  32  0.299 BFLOPs
    1 conv     64  3 x 3 / 2   416 x 416 x  32   ->   208 x 208 x  64  1.595 BFLOPs
    2 conv     32  1 x 1 / 1   208 x 208 x  64   ->   208 x 208 x  32  0.177 BFLOPs
    3 conv     64  3 x 3 / 1   208 x 208 x  32   ->   208 x 208 x  64  1.595 BFLOPs
    4 res    1                 208 x 208 x  64   ->   208 x 208 x  64
    5 conv    128  3 x 3 / 2   208 x 208 x  64   ->   104 x 104 x 128  1.595 BFLOPs
    6 conv     64  1 x 1 / 1   104 x 104 x 128   ->   104 x 104 x  64  0.177 BFLOPs
    7 conv    128  3 x 3 / 1   104 x 104 x  64   ->   104 x 104 x 128  1.595 BFLOPs
    8 res    5                 104 x 104 x 128   ->   104 x 104 x 128
    9 conv     64  1 x 1 / 1   104 x 104 x 128   ->   104 x 104 x  64  0.177 BFLOPs
   10 conv    128  3 x 3 / 1   104 x 104 x  64   ->   104 x 104 x 128  1.595 BFLOPs
   11 res    8                 104 x 104 x 128   ->   104 x 104 x 128
   12 conv    256  3 x 3 / 2   104 x 104 x 128   ->    52 x  52 x 256  1.595 BFLOPs
   13 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128  0.177 BFLOPs
   14 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256  1.595 BFLOPs
   15 res   12                  52 x  52 x 256   ->    52 x  52 x 256
   16 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128  0.177 BFLOPs
   17 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256  1.595 BFLOPs
   18 res   15                  52 x  52 x 256   ->    52 x  52 x 256
   19 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128  0.177 BFLOPs
   20 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256  1.595 BFLOPs
   21 res   18                  52 x  52 x 256   ->    52 x  52 x 256
   22 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128  0.177 BFLOPs
   23 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256  1.595 BFLOPs
   24 res   21                  52 x  52 x 256   ->    52 x  52 x 256
   25 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128  0.177 BFLOPs
   26 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256  1.595 BFLOPs
   27 res   24                  52 x  52 x 256   ->    52 x  52 x 256
   28 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128  0.177 BFLOPs
   29 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256  1.595 BFLOPs
   30 res   27                  52 x  52 x 256   ->    52 x  52 x 256
   31 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128  0.177 BFLOPs
   32 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256  1.595 BFLOPs
   33 res   30                  52 x  52 x 256   ->    52 x  52 x 256
   34 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128  0.177 BFLOPs
   35 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256  1.595 BFLOPs
   36 res   33                  52 x  52 x 256   ->    52 x  52 x 256
   37 conv    512  3 x 3 / 2    52 x  52 x 256   ->    26 x  26 x 512  1.595 BFLOPs
   38 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256  0.177 BFLOPs
   39 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512  1.595 BFLOPs
   40 res   37                  26 x  26 x 512   ->    26 x  26 x 512
   41 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256  0.177 BFLOPs
   42 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512  1.595 BFLOPs
   43 res   40                  26 x  26 x 512   ->    26 x  26 x 512
   44 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256  0.177 BFLOPs
   45 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512  1.595 BFLOPs
   46 res   43                  26 x  26 x 512   ->    26 x  26 x 512
   47 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256  0.177 BFLOPs
   48 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512  1.595 BFLOPs
   49 res   46                  26 x  26 x 512   ->    26 x  26 x 512
   50 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256  0.177 BFLOPs
   51 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512  1.595 BFLOPs
   52 res   49                  26 x  26 x 512   ->    26 x  26 x 512
   53 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256  0.177 BFLOPs
   54 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512  1.595 BFLOPs
   55 res   52                  26 x  26 x 512   ->    26 x  26 x 512
   56 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256  0.177 BFLOPs
   57 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512  1.595 BFLOPs
   58 res   55                  26 x  26 x 512   ->    26 x  26 x 512
   59 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256  0.177 BFLOPs
   60 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512  1.595 BFLOPs
   61 res   58                  26 x  26 x 512   ->    26 x  26 x 512
   62 conv   1024  3 x 3 / 2    26 x  26 x 512   ->    13 x  13 x1024  1.595 BFLOPs
   63 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512  0.177 BFLOPs
   64 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024  1.595 BFLOPs
   65 res   62                  13 x  13 x1024   ->    13 x  13 x1024
   66 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512  0.177 BFLOPs
   67 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024  1.595 BFLOPs
   68 res   65                  13 x  13 x1024   ->    13 x  13 x1024
   69 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512  0.177 BFLOPs
   70 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024  1.595 BFLOPs
   71 res   68                  13 x  13 x1024   ->    13 x  13 x1024
   72 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512  0.177 BFLOPs
   73 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024  1.595 BFLOPs
   74 res   71                  13 x  13 x1024   ->    13 x  13 x1024
   75 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512  0.177 BFLOPs
   76 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024  1.595 BFLOPs
   77 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512  0.177 BFLOPs
   78 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024  1.595 BFLOPs
   79 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512  0.177 BFLOPs
   80 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024  1.595 BFLOPs
   81 conv    255  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 255  0.088 BFLOPs
   82 detection
   83 route  79
   84 conv    256  1 x 1 / 1    13 x  13 x 512   ->    13 x  13 x 256  0.044 BFLOPs
   85 upsample            2x    13 x  13 x 256   ->    26 x  26 x 256
   86 route  85 61
   87 conv    256  1 x 1 / 1    26 x  26 x 768   ->    26 x  26 x 256  0.266 BFLOPs
   88 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512  1.595 BFLOPs
   89 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256  0.177 BFLOPs
   90 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512  1.595 BFLOPs
   91 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256  0.177 BFLOPs
   92 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512  1.595 BFLOPs
   93 conv    255  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 255  0.177 BFLOPs
   94 detection
   95 route  91
   96 conv    128  1 x 1 / 1    26 x  26 x 256   ->    26 x  26 x 128  0.044 BFLOPs
   97 upsample            2x    26 x  26 x 128   ->    52 x  52 x 128
   98 route  97 36
   99 conv    128  1 x 1 / 1    52 x  52 x 384   ->    52 x  52 x 128  0.266 BFLOPs
  100 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256  1.595 BFLOPs
  101 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128  0.177 BFLOPs
  102 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256  1.595 BFLOPs
  103 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128  0.177 BFLOPs
  104 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256  1.595 BFLOPs
  105 conv    255  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 255  0.353 BFLOPs
  106 detection
```

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I used a trained model. Below description is how I generate a pre-train model yolo.h5

- Download official [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights) and put it on top floder of project.

- Run the follow command to convert darknet weight file to keras h5 file. The `yad2k.py` was modified from [allanzelener/YAD2K](https://github.com/allanzelener/YAD2K).

```
python yad2k.py cfg\yolo.cfg yolov3.weights data\yolo.h5
```


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

YOLOv3 do not need sliding window, because its output will decide where object is (x,y) and how large box is (width, height).

as below picture

![alt text][image7]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

1. Enter a image and crop it with ROI
2. Resize the image into 416 * 416, becuase I use YOLOv3-416, that is its input size.
3. After YOLOv3 algorithm, it will output probability, tx, ty, tw, th and class1, class2, ..., calssN
4. Every pixel will be getting score with probability * ( class1, class2, ..., calssN )
5. Eliminate lower score, filtrate bounding box with non-max suppression.

![alt text][image3]
![alt text][image4]
![alt text][image5]
---

#### 1. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

First, lower score will be eliminated with a threshold, left object is our target.

![alt text][image6]



Second, there are overlapping boxes there, so we use non-max suppresion with IOU to filtrate it.
Below picture is the description about how it work. 

```
while order.size > 0:
    i = order[0]
    keep.append(i)

    xx1 = np.maximum(x[i], x[order[1:]])
    yy1 = np.maximum(y[i], y[order[1:]])
    xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
    yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

    w1 = np.maximum(0.0, xx2 - xx1 + 1)
    h1 = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w1 * h1

    ovr = inter / (areas[i] + areas[order[1:]] - inter)
    inds = np.where(ovr <= self._t2)[0]
    order = order[inds + 1]
```

![alt text][image8]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I find that the small object still hard to detect so it will detect fail when car is going futher. Maybe it need to re-train with more small object so it can get more robust result.


## Reference

	@article{YOLOv3,  
	  title={YOLOv3: An Incremental Improvement},  
	  author={J Redmon, A Farhadi },
	  year={2018}

https://pjreddie.com/media/files/papers/YOLOv3.pdf

https://github.com/allanzelener/YAD2K

http://tarangshah.com/blog/2018-01-27/what-is-map-understanding-the-statistic-of-choice-for-comparing-object-detection-models/
