"""Demo for use yolo v3
"""
import os
import time
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from model.yolo_model import YOLO
from moviepy.editor import VideoFileClip
from IPython.display import HTML
#%matplotlib inline

CROP_Y1 = 380
CROP_Y2 = 600
CROP_X1 = 650

CAR_CLASS = 2

def process_image(img):
    # It's YOLOv3-416， so we need to resize it to 416 * 416
    image = cv2.resize(img, (416, 416), interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    image /= 255.
    image = np.expand_dims(image, axis=0)

    return image


def get_classes(file):
    with open(file) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    return class_names


def draw(image, boxes, scores, classes, all_classes):
    for box, score, cl in zip(boxes, scores, classes):

        # Detect car only and ignore others calss
        if cl == CAR_CLASS:
            x, y, w, h = box

            # Need to add compensation, because I crop the image before.
            top = max(0, np.floor(x + 0.5).astype(int)) + CROP_X1
            left = max(0, np.floor(y + 0.5).astype(int)) + CROP_Y1
            right = min(image.shape[1], np.floor(x + w + 0.5).astype(int)) + CROP_X1
            bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int)) + CROP_Y1

            cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
            cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 1,
                    cv2.LINE_AA)


def detect_image(image, yolo, all_classes):

    # Crop image for ROI
    img  = image[CROP_Y1:CROP_Y2, CROP_X1:]
    pimage = process_image(img)

    start = time.time()
    boxes, classes, scores = yolo.predict(pimage, img.shape)
    end = time.time()

    print('time: {0:.2f}s'.format(end - start))

    if boxes is not None:
        draw(image, boxes, scores, classes, all_classes)

    return image


def Vehicle_Detection(image):
    return detect_image(image, yolo, all_classes)


project_video_output = './project_video_output.mp4'
clip1 = VideoFileClip("./project_video.mp4")

yolo = YOLO(0.6, 0.5)
file = 'data/coco_classes.txt'
all_classes = get_classes(file)

#ret_clip = clip1.fl_image(Vehicle_Detection).subclip(35, None)
ret_clip = clip1.fl_image(Vehicle_Detection)
ret_clip.write_videofile(project_video_output, audio=False)
