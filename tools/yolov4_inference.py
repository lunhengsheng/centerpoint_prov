#!/usr/bin/env python3
import rospy
import ros_numpy
import numpy as np
import copy
import json
import os
import sys
import time
import darknet

import cv2

from cv_bridge import CvBridge, CvBridgeError


from std_msgs.msg import Header

from sensor_msgs.msg import Image

from darknet_ros_msgs.msg import BoundingBox2D, BoundingBoxes2D


CV_BRIDGE = CvBridge()



def bbox2points(bbox):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """
    x, y, w, h = bbox
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def image_detection(image, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect

    imheight, imwidth, _ = image.shape

    darknet_image = darknet.make_image(imwidth, imheight, 3)

    darknet.copy_image_from_bytes(darknet_image, image.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    #print("detections: ", detections)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image, class_colors)
    return image, detections





class Processor_ROS:
    def __init__(self, _2d_config_path, _2d_weights_path, _2d_data_file):
        self._2d_config_path = _2d_config_path
        self._2d_weights_path = _2d_weights_path
        self._2d_data_file = _2d_data_file
        self.yolov4_detector = None
        self.class_names = None
        self.class_colors = None
       
    def initialize(self):
        self.read_config()
        
    def read_config(self):
        self.yolov4_detector, self.class_names, self.class_colors = darknet.load_network(
            self._2d_config_path,
            self._2d_data_file,
            self._2d_weights_path ,
            batch_size=1
        )

    def detect_2d(self, image_input):

        t1 = time.time()

        image_output, detections = image_detection(
            image_input, self.yolov4_detector, self.class_names, self.class_colors, 0.3
        )

        print("yolov4 inference time is ", time.time()-t1)
        darknet.print_detections(detections, True)
        return image_output, detections






def image_callback(input_imgmsg):
    t_t = time.time()

    cv_img = CV_BRIDGE.imgmsg_to_cv2(input_imgmsg, 'bgr8')

    img_with_2d_boxes, detections = proc_1.detect_2d(cv_img)

    output_imgmsg = CV_BRIDGE.cv2_to_imgmsg(img_with_2d_boxes, "bgr8")

    output_imgmsg.header.stamp = input_imgmsg.header.stamp
    output_imgmsg.header.frame_id = input_imgmsg.header.frame_id

    image_pub.publish(output_imgmsg)

    arr_bbox = BoundingBoxes2D()

    arr_bbox.header.stamp = input_imgmsg.header.stamp
    arr_bbox.header.frame_id = input_imgmsg.header.frame_id

    for label, confidence, bbox in detections:
        box = BoundingBox2D()
        left, top, right, bottom = bbox2points(bbox)
        box.probability = float(confidence)
        box.xmin = left
        box.ymin = top
        box.xmax = right
        box.ymax = bottom
        box.Class = label
        arr_bbox.bounding_boxes.append(box)

    if len(arr_bbox.bounding_boxes) is not 0:
        #rospy.loginfo(arr_bbox)
        detections_pub.publish(arr_bbox)
        arr_bbox.bounding_boxes = []
    else:
        arr_bbox.bounding_boxes = []
        detections_pub.publish(arr_bbox)



   
if __name__ == "__main__":

    global proc

    rospy.init_node('yolov4_ros_node')

    _2d_config_path = '/root/darknet/cfg/custom-yolov4-detector-kitti_all_classes.cfg'
    _2d_weights_path = '/root/darknet/backup/custom-yolov4-detector-kitti_all_classes_best.weights'
    _2d_data_file = '/root/darknet/cfg/coco.data'

    proc_1 = Processor_ROS(_2d_config_path, _2d_weights_path, _2d_data_file)

    proc_1.initialize()
    

    image_topic = "/kitti/camera_color_left/image_raw"

    image_sub = rospy.Subscriber(image_topic, Image, image_callback)

    image_pub = rospy.Publisher("image_with_2D_detections", Image, queue_size=1)

    detections_pub = rospy.Publisher("detection_2D_boxes", BoundingBoxes2D, queue_size=1)

        

    print("[+] YOLOv4 ros_node has started!")    
    rospy.spin()
