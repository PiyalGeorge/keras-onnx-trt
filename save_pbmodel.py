import tensorflow as tf
from core.yolov4 import YOLO, decode, filter_boxes
import core.utils as utils

def main():
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config()
    input_size = 416
    inputweight = './data/yolov4.weights'
    outputweight = './output/yolov4pb'
    input_layer = tf.keras.layers.Input([input_size, input_size, 3])
    feature_maps = YOLO(input_layer, NUM_CLASS)
    bbox_tensors = []
    prob_tensors = []
    for i, fm in enumerate(feature_maps):
        if i == 0:
            output_tensors = decode(fm, input_size // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
        elif i == 1:
            output_tensors = decode(fm, input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
        else:
            output_tensors = decode(fm, input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
        bbox_tensors.append(output_tensors[0])
        prob_tensors.append(output_tensors[1])
    pred_bbox = tf.concat(bbox_tensors, axis=1)
    pred_prob = tf.concat(prob_tensors, axis=1)
    boxes, pred_conf = filter_boxes(pred_bbox, pred_prob, score_threshold=.4, input_shape=tf.constant([input_size, input_size]))
    pred = tf.concat([boxes, pred_conf], axis=-1)
    model = tf.keras.Model(input_layer, pred)
    utils.load_weights(model, inputweight)
    model.summary()
    model.save(outputweight)

if __name__ == "__main__":
    main()

