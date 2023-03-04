#!/usr/bin/env python3
"""Contain the class 'Yolo'"""

import numpy as np
import tensorflow.keras as K


def sigmoid(number):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-number))


class Yolo:
    """Uses the Yolo v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize the class Yolo

        Args:
            model_path: is the path to where a Darknet Keras model is stored
            classes_path: is the path to where the list of class names used for
                  the Darknet model, listed in order of index, can be found
            class_t: representing the box score threshold for the initial
                     filtering step
            nms_t: representing the IOU threshold for non-max suppression
            anchors:(outputs,anchor_boxes,2)containing all of the anchor boxs
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'rt') as file:
            self.class_names = file.read().rstrip('\n').split('\n')
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Args:
            outputs: contains the predictions from the Darknet
                model for a single image
            image_size: contains the image's original size
        """
        h, w = image_size[0], image_size[1]
        boxes = [output[..., :4] for output in outputs]
        confidence, probs = [], []
        cornerX, cornerY = [], []

        for output in outputs:
            gridH, gridW, anchors = output.shape[:3]
            cx = np.arange(gridW).reshape(1, gridW)
            cx = np.repeat(cx, gridH, axis=0)
            cy = np.arange(gridW).reshape(1, gridW)
            cy = np.repeat(cy, gridH, axis=0).T

            cornerX.append(np.repeat(cx[..., np.newaxis], anchors, axis=2))
            cornerY.append(np.repeat(cy[..., np.newaxis], anchors, axis=2))
            confidence.append(sigmoid(output[..., 4:5]))
            probs.append(sigmoid(output[..., 5:]))

        inputW = self.model.input.shape[1]
        inputH = self.model.input.shape[2]

        for x, box in enumerate(boxes):
            bx = ((sigmoid(box[..., 0])+cornerX[x])/outputs[x].shape[1])
            by = ((sigmoid(box[..., 1])+cornerY[x])/outputs[x].shape[0])
            bw = ((np.exp(box[..., 2])*self.anchors[x, :, 0])/inputW)
            bh = ((np.exp(box[..., 3])*self.anchors[x, :, 1])/inputH)

            box[..., 0] = (bx - (bw * 0.5))*w
            box[..., 1] = (by - (bh * 0.5))*h
            box[..., 2] = (bx + (bw * 0.5))*w
            box[..., 3] = (by + (bh * 0.5))*h

        return (boxes, confidence, probs)
