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


import cv2
import message_filters

from cv_bridge import CvBridge, CvBridgeError


from std_msgs.msg import Header, ColorRGBA
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField, Image
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from geometry_msgs.msg import Quaternion, Pose, Point, Vector3
from darknet_ros_msgs.msg import BoundingBox2D, BoundingBoxes2D
from pyquaternion import Quaternion as Pyquaternion

from visualization_msgs.msg import Marker, MarkerArray


from det3d.models import build_fusion
from det3d.torchie import Config
from det3d.core.input.voxel_generator import VoxelGenerator

from det3d.core.bbox import box_np_ops

from queue import Queue



try:
    from apex import amp
except:
    print("No APEX!")


CV_BRIDGE = CvBridge()


rect = np.array([[ 9.999478e-01,   9.791707e-03, -2.925305e-03,  0.        ],
 [-9.806939e-03,  9.999382e-01,  -5.238719e-03,  0.        ],
 [ 2.873828e-03,  5.267134e-03,  9.999820e-01,   0.        ],
 [ 0.,          0.,          0.,          1.        ]])
Trv2c = np.array([[ 7.755449e-03, -9.999694e-01,  -1.014303e-03, -7.275538e-03],
 [2.294056e-03,  1.032122e-03, -9.999968e-01,  -6.324057e-02],
 [ 9.999673e-01,   7.753097e-03, 2.301990e-03,  -2.670414e-01 ],
 [ 0.,          0.,          0.,          1.        ]])
P2 = np.array([[ 7.183351e+02,  0.000000e+00,  6.003891e+02,  4.450382e+01],
 [ 0.000000e+00,  7.183351e+02,  1.815122e+02, -5.951107e-01],
 [ 0.000000e+00,  0.000000e+00,  1.000000e+00,  2.616315e-03],
 [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]])



def draw_projected_box3d(image, qs, label, score, minxy, width, height, color=(0, 255, 0), thickness=2):
    """ Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    """
    qs = qs.astype(np.int32)

    cv2.putText(image, "{} [{:.2f}]".format(label, score),
                (int(minxy[0]), int(minxy[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 2)
                
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        # cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)
        ret, p1, p2 = cv2.clipLine((0, 0, width, height),  (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]))


        cv2.line(image, p1 , p2 , color, thickness)


        i, j = k + 4, (k + 1) % 4 + 4
        ret, p3, p4 = cv2.clipLine((0, 0, width, height),  (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]))


        cv2.line(image, p3 , p4 , color, thickness)

        i, j = k, k + 4
        ret, p5, p6 = cv2.clipLine((0, 0, width, height),  (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]))


        cv2.line(image, p5 , p6 , color, thickness)


    return image


def yaw2quaternion(yaw: float) -> Pyquaternion:
    return Pyquaternion(axis=[0,0,1], radians=yaw)

def get_annotations_indices(types, thresh, label_preds, scores):
    indexs = []
    annotation_indices = []
    for i in range(label_preds.shape[0]):
        if label_preds[i] == types:
            indexs.append(i)
    for index in indexs:
        if scores[index] >= thresh:
            annotation_indices.append(index)
    return annotation_indices  


def remove_low_score_nu(image_anno, thresh):
    img_filtered_annotations = {}
    label_preds_ = image_anno["label_preds"].detach().cpu().numpy()
    scores_ = image_anno["scores"].detach().cpu().numpy()
    
    car_indices =                  get_annotations_indices(0, 0.4, label_preds_, scores_)
    pedestrian_indices =              get_annotations_indices(1, 0.4, label_preds_, scores_)
    cyclist_indices =              get_annotations_indices(2, 0.4, label_preds_, scores_)
    van_indices =              get_annotations_indices(3, 0.4, label_preds_, scores_)




    for key in image_anno.keys():
        if key == 'metadata':
            continue
        img_filtered_annotations[key] = (
            image_anno[key][car_indices +
                            pedestrian_indices +
                            cyclist_indices +
                            van_indices
                            # truck_indices
                            ])

    return img_filtered_annotations


class Processor_ROS:
    def __init__(self, config_path, model_path):
        self.points = None
        self.config_path = config_path
        self.model_path = model_path
        self.device = None
        self.fusion = None
        self.voxel_generator = None
        self.inputs = None
        self.q = Queue(maxsize = 1)



        
    def initialize(self):
        self.read_config()
        
    def read_config(self):
        config_path = self.config_path
        cfg = Config.fromfile(self.config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fusion = build_fusion(cfg.fusion)
        checkpoint = torch.load(self.model_path)



        self.fusion.load_state_dict(checkpoint["state_dict"])
        
        self.fusion = self.fusion.half()


        self.fusion = self.fusion.to(self.device).eval()


        self.range = cfg.voxel_generator.range
        self.voxel_size = cfg.voxel_generator.voxel_size
        self.max_points_in_voxel = cfg.voxel_generator.max_points_in_voxel
        self.max_voxel_num = cfg.voxel_generator.max_voxel_num
        self.voxel_generator = VoxelGenerator(
            voxel_size=self.voxel_size,
            point_cloud_range=self.range,
            max_num_points=self.max_points_in_voxel,
            max_voxels=self.max_voxel_num[0],
        )

    def run_3d_prediction(self, points, points_stamp):
        t_t = time.time()
        print(f"input points shape: {points.shape}")
        num_features = 4   
        self.points = points.reshape([-1, num_features])



        #self.points[:, 4] = 0 # timestamp value 

        print("points shape: ", self.points.shape)
        
        voxels, coords, num_points = self.voxel_generator.generate(self.points)
        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)

        grid_size = self.voxel_generator.grid_size
        coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values = 0)
        self.points = np.pad(self.points, ((0, 0), (1, 0)), mode='constant', constant_values = 0)
        voxels = torch.tensor(voxels, dtype=torch.float32, device=self.device)
        coords = torch.tensor(coords, dtype=torch.int32, device=self.device)
        num_points = torch.tensor(num_points, dtype=torch.int32, device=self.device)
        num_voxels = torch.tensor(num_voxels, dtype=torch.int32, device=self.device)
        points = torch.tensor(self.points, dtype=torch.float32, device=self.device)

        voxels = voxels.half()
        
        
        self.inputs = dict(
            points = points,
            voxels = voxels,
            num_points = num_points,
            num_voxels = num_voxels,
            coordinates = coords,
            shape = [grid_size]
        )

        t = time.time()

        torch.cuda.synchronize()
        with torch.no_grad():
        
            preds_dict = self.fusion._3d_detector(self.inputs, return_loss=False, pre_fusion=True)[0]

        torch.cuda.synchronize()
        print("3D predictions time cost:", time.time() - t)

        _3d_preds = dict(
            preds_dict = preds_dict,
            stamp = points_stamp
        )


        return _3d_preds

        

        

    def run_fusion(self, predictions3D, image_shape, predictions2D):
        t_t = time.time()


        online_data = {}

        calib = {"rect": torch.from_numpy(rect).cuda(),
                 "Trv2c": torch.from_numpy(Trv2c).cuda(),
                 "P2": torch.from_numpy(P2).cuda()}
        
        online_data.update({"predictions3D": predictions3D, "predictions2D": predictions2D, "calib": calib, "image_shape": image_shape})


        torch.cuda.synchronize()
        t = time.time()

        with torch.no_grad():
            outputs = self.fusion(None, return_loss=False, online_data=online_data)[0]

        
    
        print("outputs before filter: ", len(outputs))
        
        torch.cuda.synchronize()
        print("  network predict time cost:", time.time() - t)

        outputs = remove_low_score_nu(outputs, 0.45)

        print("outputs after filter: ", len(outputs))

        print(outputs)

        boxes_lidar = outputs["box3d_lidar"].detach().cpu().numpy()
        print("  predict boxes:", boxes_lidar.shape)

        scores = outputs["scores"].detach().cpu().numpy()
        types = outputs["label_preds"].detach().cpu().numpy()

        boxes_lidar[:, -1] = -boxes_lidar[:, -1] #- np.pi / 2

        print(f"  total cost time: {time.time() - t_t}")
  
        return scores, boxes_lidar, types

    

def get_xyz_points(cloud_array, remove_nans=True, dtype=np.float):
    '''
    '''
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]



    points = np.zeros(cloud_array.shape + (4,), dtype=dtype)
    points[...,0] = cloud_array['x']
    points[...,1] = cloud_array['y']
    points[...,2] = cloud_array['z'] 
    points[...,3] = cloud_array['i']
    #points[...,4] = cloud_array['reflectivity'].astype("float32")

    #print("points.shape ", points.shape)
    return points

def xyz_array_to_pointcloud2(points_sum, stamp=None, frame_id=None):
    '''
    Create a sensor_msgs.PointCloud2 from an array of points.
    '''
    msg = PointCloud2()
    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = points_sum.shape[0]
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('i', 12, PointField.FLOAT32, 1)
        ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = points_sum.shape[0]
    msg.is_dense = int(np.isfinite(points_sum).all())
    msg.data = np.asarray(points_sum, np.float32).tostring()
    return msg


callback_times = []
num_frames = 600

def parse_boxes_2D_msg(boxes2D_msg):
    predictions_car = np.empty((0,6))
    predictions_ped = np.empty((0,6))
    predictions_cyc = np.empty((0,6))
    predictions_van = np.empty((0,6))
    # predictions_truck = np.empty((0,6))


    for box2D in boxes2D_msg.bounding_boxes:
        class_name = box2D.Class
        xmin = box2D.xmin
        ymin = box2D.ymin
        xmax = box2D.xmax
        ymax = box2D.ymax
        score = box2D.probability/100
        temp = np.array([class_name, xmin, ymin, xmax, ymax, score], dtype=object)

        if class_name == 'Car':
            # continue
            predictions_car = np.vstack((predictions_car,temp))
        elif class_name == 'Pedestrian':
            # pass
            predictions_ped = np.vstack((predictions_ped, temp))
        elif class_name == 'Cyclist':
            # pass
            predictions_cyc = np.vstack((predictions_cyc, temp))
        elif class_name == 'Van':
            # continue
            predictions_van = np.vstack((predictions_van, temp))
        # elif class_name == 'Truck':
        #     predictions_truck = np.vstack((predictions_truck, temp))



    predictions = [predictions_car, predictions_ped, predictions_cyc, predictions_van]

    print("predictions 2D: ", predictions)

    return predictions


def draw_3d_boxes_on_image(boxes_lidar, types, scores, img, image_shape):

    boxes_lidar[:, -1] = -boxes_lidar[:, -1] #- np.pi / 2


    final_box_preds_camera = box_np_ops.box_lidar_to_camera(boxes_lidar, rect, Trv2c)
    locs = final_box_preds_camera[:, :3]
    dims = final_box_preds_camera[:, 3:6]
    angles = final_box_preds_camera[:, 6]
    camera_box_origin = [0.5, 0.5, 0.5]

    boxes_corners = box_np_ops.center_to_corner_box3d(
        locs, dims, angles, camera_box_origin, axis=1)

    boxes3d_pts_2d = box_np_ops.project_to_image(
        boxes_corners, P2)

    height, width = image_shape

    for i in range(boxes3d_pts_2d.shape[0]):

        box3d_pts_2d = boxes3d_pts_2d[i]

        #print("image size: (height, width)", (height, width))



        minxy = np.amin(box3d_pts_2d, axis=0)
        maxxy = np.amax(box3d_pts_2d, axis=0)


        if (minxy[0] < 0 and minxy[1]) < 0 or  (maxxy[0] < 0 and maxxy[1] < 0):
            print("minxy: ", minxy)
            continue
        elif (minxy[0] > width and minxy[1] > height) or (maxxy[0] > width and maxxy[1] > height):
            print("maxy: ", maxxy)
            continue

        else:

            if types[i] == 0:
                img = draw_projected_box3d(img, box3d_pts_2d, "Car", scores[i], minxy, width, height, color=(0, 255, 0))
            elif types[i] == 1:
                img = draw_projected_box3d(img, box3d_pts_2d, "Pedestrian", scores[i], minxy, width, height, color=(255, 255, 0))
            elif types[i] == 2:
                img = draw_projected_box3d(img, box3d_pts_2d, "Cyclist", scores[i], minxy, width, height, color=(0, 255, 255))
            elif types[i] == 3:
                img = draw_projected_box3d(img, box3d_pts_2d, "Van", scores[i], minxy, width, height, color=(0, 128, 255))
            # elif types[i] == 4:
            #     img = draw_projected_box3d(img, box3d_pts_2d, minxy, width, height, color=(0, 0, 255))
            # elif types[i] == 5:
            #     img = draw_projected_box3d(img, box3d_pts_2d, minxy, width, height, color=(255, 0, 0))

    return img



def fusion_callback(img_msg, pc_msg, boxes2D_msg):

    print("image timestamp in fusion callback: ", img_msg.header.stamp.secs, img_msg.header.stamp.nsecs)
    print("point cloud timestamp in fusion callback: ", pc_msg.header.stamp.secs, pc_msg.header.stamp.nsecs)
    print("2D boxes timestamp in fusion callback: ", boxes2D_msg.header.stamp.secs, boxes2D_msg.header.stamp.nsecs)

    print("Queue is empty in fusion?: ", proc_1.q.empty())
    print("Queue is full in fusion?: ", proc_1.q.full())

    _3d_preds = proc_1.q.get()
    predictions3D = _3d_preds['preds_dict']
    predictions3D_stamp = _3d_preds['stamp']
    print("3D initial boxes timestamp in fusion callback: ", predictions3D_stamp.secs, predictions3D_stamp.nsecs)

    cv_img = CV_BRIDGE.imgmsg_to_cv2(img_msg, 'bgr8')

    predictions2D = parse_boxes_2D_msg(boxes2D_msg)


    image_shape = cv_img.shape[:2]
    scores, boxes_lidar, types = proc_1.run_fusion(predictions3D, image_shape, predictions2D)

    arr_bbox = BoundingBoxArray()
    arr_marker = MarkerArray()

    if scores.size != 0:
        for i in range(scores.size):
            bbox = BoundingBox()
            bbox.header.frame_id = pc_msg.header.frame_id
            bbox.header.stamp = pc_msg.header.stamp
            q = yaw2quaternion(float(boxes_lidar[i][-1]))
            bbox.pose.orientation.x = q[1]
            bbox.pose.orientation.y = q[2]
            bbox.pose.orientation.z = q[3]
            bbox.pose.orientation.w = q[0]           
            bbox.pose.position.x = float(boxes_lidar[i][0])
            bbox.pose.position.y = float(boxes_lidar[i][1])
            bbox.pose.position.z = float(boxes_lidar[i][2])
            bbox.dimensions.x = float(boxes_lidar[i][3])
            bbox.dimensions.y = float(boxes_lidar[i][4])
            bbox.dimensions.z = float(boxes_lidar[i][5])
            bbox.value = scores[i]
            bbox.label = int(types[i])
            arr_bbox.boxes.append(bbox)

            if types[i] == 0:
                text = "{} [{:.2f}]".format("Car", scores[i])
                color_label = ColorRGBA(0.0, 1.0, 0.0, 1.0)
            elif types[i] == 1:
                text = "{} [{:.2f}]".format("Pedestrian", scores[i])
                color_label = ColorRGBA(0.0, 1.0, 1.0, 1.0)
            elif types[i] == 2:
                text = "{} [{:.2f}]".format("Cyclist", scores[i])
                color_label = ColorRGBA(1.0, 1.0, 0.0, 1.0)
            elif types[i] == 3:
                text = "{} [{:.2f}]".format("Van", scores[i])
                color_label = ColorRGBA(1.0, 0.501960, 0.0, 1.0)

            text_marker = Marker(
                type=Marker.TEXT_VIEW_FACING,
                ns = "frame_" + str(i),
                id=i,
                lifetime=rospy.Duration(0.1),
                pose=Pose(Point(bbox.pose.position.x, bbox.pose.position.y, bbox.pose.position.z + bbox.dimensions.z/2 + 0.5), 
                    Quaternion(bbox.pose.orientation.x, bbox.pose.orientation.y, bbox.pose.orientation.z, bbox.pose.orientation.w)),
                scale=Vector3(0.7, 0.7, 0.7),
                header=Header(frame_id=bbox.header.frame_id),
                color=color_label,
                text=text)
            
            arr_marker.markers.append(text_marker)


    arr_bbox.header.frame_id = pc_msg.header.frame_id
    arr_bbox.header.stamp = pc_msg.header.stamp
    if len(arr_bbox.boxes) is not 0:
        pub_arr_bbox.publish(arr_bbox)
        pub_arr_marker.publish(arr_marker)
        arr_bbox.boxes = []
        arr_marker.markers = []
    else:
        arr_bbox.boxes = []
        arr_marker.markers = []
        pub_arr_bbox.publish(arr_bbox)
        pub_arr_marker.publish(arr_marker)
        

    
    img_with_3d_boxes = draw_3d_boxes_on_image(boxes_lidar, types, scores, cv_img, image_shape)

    image_pub.publish(CV_BRIDGE.cv2_to_imgmsg(img_with_3d_boxes, "bgr8"))

    # print("total callback time: ", callback_time)


    # callback_times.append(callback_time)

    # print("callback_times length: ", len(callback_times))

    # if len(callback_times) == num_frames:
    #     print("Average inference speed over {} frames is {:f} ms".format(num_frames, np.mean(np.asarray(callback_times))*1000))
    #     callback_times.clear()


def lidar_callback(img_msg, pc_msg):

    print("image timestamp in lidar callback: ", img_msg.header.stamp.secs, img_msg.header.stamp.nsecs)
    print("point cloud timestamp in lidar callback: ", pc_msg.header.stamp.secs, pc_msg.header.stamp.nsecs)

    msg_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(pc_msg)


    points_stamp = pc_msg.header.stamp

    points = get_xyz_points(msg_cloud, True)
    print("  ")

    _3d_preds = proc_1.run_3d_prediction(points, points_stamp)

    if proc_1.q.full():
        proc_1.q.get()
        print("Element deleted")
        print("Queue is empty in lidar?: ", proc_1.q.empty())



    if proc_1.q.empty():
        proc_1.q.put(_3d_preds)
        print("Element inserted")
        print("Queue is full in lidar?: ", proc_1.q.full())



   
if __name__ == "__main__":

    global proc
    ## CenterPoint
    config_path = 'configs/kitti_clocs_four_heads_car_ped_cyc_van.py'
    model_path = 'models/kitti_clocs_four_heads_car_ped_cyc_van/epoch_20.pth'

    proc_1 = Processor_ROS(config_path, model_path)
    
    proc_1.initialize()
    
    rospy.init_node('centerpoint_clocs_ros_node')
    sub_lidar_topic = [ "/velodyne_points", 
                        "/top/rslidar_points",
                        "/points_raw", 
                        "/lidar_protector/merged_cloud", 
                        "/merged_cloud",
                        "/lidar_top", 
                        "/roi_pclouds",
                        "/os_cloud_node/points",
                        "/custom_pc",
                        "/scala_gen2_points",
                        "/kitti/velo/pointcloud"]

    image_topic = "/kitti/camera_color_left/image_raw"

    boxes_2D_topic = "/detection_2D_boxes"

    image_sub = message_filters.Subscriber(image_topic, Image)

    image_sub2 = message_filters.Subscriber(image_topic, Image)

            
    
    pointcloud_sub = message_filters.Subscriber(sub_lidar_topic[10], PointCloud2)
    pointcloud_sub2 = message_filters.Subscriber(sub_lidar_topic[10], PointCloud2)

    boxes_2D_sub = message_filters.Subscriber(boxes_2D_topic, BoundingBoxes2D)
    
    pub_arr_bbox = rospy.Publisher("detection_3D_boxes_fusion", BoundingBoxArray, queue_size=1)

    pub_arr_marker = rospy.Publisher("3D_labels_fusion", MarkerArray, queue_size=1)
    image_pub = rospy.Publisher("image_with_3D_detections_fusion", Image, queue_size=1)


    ats1 = message_filters.ApproximateTimeSynchronizer(
        [image_sub, pointcloud_sub], queue_size=1, slop=0.1)
    ats1.registerCallback(lidar_callback)   

    ats2 = message_filters.ApproximateTimeSynchronizer(
        [image_sub2, pointcloud_sub2, boxes_2D_sub], queue_size=1, slop=0.1)
    ats2.registerCallback(fusion_callback)
        

    print("[+] CenterPoint + CLOCs ros_node has started!")    
    rospy.spin()
