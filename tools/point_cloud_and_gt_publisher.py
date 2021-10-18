#!/usr/bin/env python3
import rospy
import ros_numpy
import numpy as np
import copy
import json
import os
import sys
import torch
import time 
import pickle
import itertools

from std_msgs.msg import Header, ColorRGBA
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from visualization_msgs.msg import Marker, MarkerArray


from geometry_msgs.msg import Quaternion, Pose, Point, Vector3


from pyquaternion import Quaternion

from pypcd import pypcd



from det3d.models import build_detector
from det3d.torchie import Config
from det3d.core.input.voxel_generator import VoxelGenerator


def load_pcd(path):
    cloud = pypcd.PointCloud.from_path(path)

    fake_timestamp = np.zeros((cloud.pc_data['t'].shape), dtype=np.float32)
    cloud = pypcd.update_field(cloud, "t", fake_timestamp)

    points = np.column_stack((cloud.pc_data['x'],
                        cloud.pc_data['y'], 
                        cloud.pc_data['z'],
                        cloud.pc_data['intensity'],
                        cloud.pc_data['reflectivity'].astype('float32')
    ))

    #points = np.append(points, fake_timestamp, axis=1)

    print("points.shape: ", points.shape)

    
    return points


def load_info_file(info_path):
    dbfile = open(info_path, 'rb')
    db = pickle.load(dbfile)
    dbfile.close()
    return db




def yaw2quaternion(yaw: float) -> Quaternion:
    return Quaternion(axis=[0,0,1], radians=yaw)


def xyzi_array_to_pointcloud2(pcd_path, stamp=None, frame_id=None):
    '''
    Create a sensor_msgs.PointCloud2 from an array of points.
    
    '''

    points = load_pcd(pcd_path)
    
    msg = PointCloud2()
    if stamp:
        msg.header.stamp = rospy.Time.now()
    if frame_id:
        msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = points.shape[0]
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('intensity', 12, PointField.FLOAT32, 1),
        PointField('reflectivity', 16, PointField.FLOAT32, 1)
        ]
    msg.is_bigendian = False
    msg.point_step = 20
    msg.row_step = points.shape[0]
    msg.is_dense = int(np.isfinite(points).all())
    msg.data = np.asarray(points, np.float32).tostring()

    #print("msg.data", msg.data)

    return msg

def publish_all(lidar_path, frame_id):

    LABELS = ["car", "trailer", "bus", "van", "pedestrian"]

    for pcd_file in sorted(os.listdir(lidar_path)):

        point_cloud = PointCloud2()

        point_cloud = xyzi_array_to_pointcloud2(lidar_path + pcd_file, frame_id = frame_id)

        # gt_arr_bbox = BoundingBoxArray()
        # gt_arr_marker = MarkerArray()

        # gt_boxes_in_frame = database[i]["gt_boxes"]
        # gt_names_in_frame = database[i]["gt_names"]

        # if gt_boxes_in_frame.size !=0:

        #     for j in range (gt_boxes_in_frame.shape[0]):
        #         bbox = BoundingBox()
        #         bbox.header.frame_id = frame_id
        #         bbox.header.stamp = point_cloud.header.stamp
        #         q = yaw2quaternion(float(-gt_boxes_in_frame[j][8] - np.pi/2))
        #         bbox.pose.orientation.x = q[1]
        #         bbox.pose.orientation.y = q[2]
        #         bbox.pose.orientation.z = q[3]
        #         bbox.pose.orientation.w = q[0]           
        #         bbox.pose.position.x = float(gt_boxes_in_frame[j][0])
        #         bbox.pose.position.y = float(gt_boxes_in_frame[j][1])
        #         bbox.pose.position.z = float(gt_boxes_in_frame[j][2])
        #         bbox.dimensions.x = float(gt_boxes_in_frame[j][3])
        #         bbox.dimensions.y = float(gt_boxes_in_frame[j][4])
        #         bbox.dimensions.z = float(gt_boxes_in_frame[j][5])
        #         bbox.value = 1
        #         bbox.label = int(LABELS.index(gt_names_in_frame[j]))
        #         gt_arr_bbox.boxes.append(bbox)

        #         text_marker = Marker(
        #             type=Marker.TEXT_VIEW_FACING,
        #             ns = "frame_" + str(i),
        #             id=j,
        #             lifetime=rospy.Duration(1.5),
        #             pose=Pose(Point(bbox.pose.position.x, bbox.pose.position.y, bbox.pose.position.z + bbox.dimensions.z/2 + 0.5), 
        #                 Quaternion(bbox.pose.orientation.x, bbox.pose.orientation.y, bbox.pose.orientation.z, bbox.pose.orientation.w)),
        #             scale=Vector3(1, 1, 1),
        #             header=Header(frame_id=frame_id),
        #             color=ColorRGBA(1.0, 1.0, 1.0, 0.8),
        #             text=gt_names_in_frame[j])
                
        #         gt_arr_marker.markers.append(text_marker)


        pc_publisher.publish(point_cloud)

        # gt_arr_bbox.header.frame_id = frame_id
        # gt_arr_bbox.header.stamp = point_cloud.header.stamp



        # if len(gt_arr_bbox.boxes) is not 0:
        #     gt_box_publisher.publish(gt_arr_bbox)
        #     gt_labels_publisher.publish(gt_arr_marker)
        #     gt_arr_bbox.boxes = []
        #     gt_arr_marker.markers = []
        # else:
        #     gt_arr_bbox.boxes = []
        #     gt_arr_marker.markers = []
        #     gt_box_publisher.publish(gt_arr_bbox)
        #     gt_labels_publisher.publish(gt_arr_marker)

   
if __name__ == "__main__":


    global proc

    
    rospy.init_node('point_cloud_publisher_ros_node')

    print("point cloud publisher node has started!") 

    
    pc_publisher = rospy.Publisher("custom_pc", PointCloud2, queue_size=1)
    #gt_box_publisher = rospy.Publisher("gt_boxes", BoundingBoxArray, queue_size=1)
    #gt_labels_publisher = rospy.Publisher("gt_labels", MarkerArray, queue_size=1)

    lidar_path = "/media/hdd_4tb/xavier/proanno/input/providentia/s50_s_cam_near/2021_03_26_13_s50_s_cam_near/pointclouds/" 

    while True:
        publish_all(lidar_path, frame_id = "map")
 

    rospy.spin()
