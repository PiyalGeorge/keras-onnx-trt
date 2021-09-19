import os
import tensorflow as tf
import numpy as np
import cv2
from core.yolov4 import YOLOv4, decode
import core.utils as utils
from core.config import cfg


def main():
    input_size = 416
    inputweight = './data/yolov4.weights'
    outputweight = './output/yolov4.h5'
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config()
    NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
    input_layer = tf.keras.layers.Input([input_size, input_size, 3])
    feature_maps = YOLOv4(input_layer, NUM_CLASS)
    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        print(i, fm, NUM_CLASS)
        bbox_tensor = decode(fm, input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
        bbox_tensors.append(bbox_tensor)
    model = tf.keras.Model(input_layer, bbox_tensors)
    utils.load_weights(model, inputweight)
    model = tf.keras.Model(input_layer, bbox_tensors)
    model.summary()
    utils.load_weights(model, inputweight)
    model.save(outputweight)

if __name__ == "__main__":
    main()

