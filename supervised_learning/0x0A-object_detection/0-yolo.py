#!/usr/bin/env python3
"""Contain the class 'Yolo'"""

import numpy as np
import tensorflow.keras as K


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
