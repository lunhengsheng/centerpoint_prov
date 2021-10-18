           
from os import terminal_size
from det3d.ops.nms.nms_gpu import iou_device
import io
from types import DynamicClassAttribute
from det3d.datasets.kitti import kitti_common as kitti

from det3d.datasets.utils import eval

from det3d.core.bbox import box_torch_ops, box_np_ops

import numpy as np

import time
import pathlib

import torch
           

def parse_offline_2D_detections(example, _2d_stored_detections_path):


        print("example[image] = ", example["image"])

        img_idx = example['image'][0]['image_idx']



        
        detection_2d_result_path = pathlib.Path(_2d_stored_detections_path)
        detection_2d_file_name = f"{detection_2d_result_path}/{kitti.get_image_index_str(img_idx)}.txt"


        with open(detection_2d_file_name, 'r') as f:
            lines = f.readlines()
        
        content = [line.strip().split(' ') for line in lines]
        predicted_class = np.array([x[0] for x in content],dtype='object')
        detection_result = np.array([[float(info) for info in x[1:5]] for x in content]).reshape(-1, 4)
        #print("detection result: ", detection_result)
        score = np.array([float(x[5]) for x in content])  # 1000 is the score scale!!!
        f_detection_result=np.hstack((predicted_class.reshape(-1,1), detection_result))
        #print("f_detection_result: ", f_detection_result)

        f_detection_result=np.append(f_detection_result,score.reshape(-1,1),1)
        top_predictions=f_detection_result.reshape(-1,6)
        #print("middle_predictions: ", middle_predictions)
        #top_predictions=middle_predictions[np.where(middle_predictions[:,5]>=0.4)]





        top_predictions_car = top_predictions[np.where(top_predictions[:,0]=='Car')]

        top_predictions_ped = top_predictions[np.where(top_predictions[:,0]=='Pedestrian')]
        top_predictions_cyc = top_predictions[np.where(top_predictions[:,0]=='Cyclist')]
        top_predictions_van = top_predictions[np.where(top_predictions[:,0]=='Van')]
        # top_predictions_truck = top_predictions[np.where(top_predictions[:,0]=='Truck')]



        # print("top_predictions_car shape: ", top_predictions_car.shape)

        # print("top_predictions_ped shaoe: ", top_predictions_ped.shape)
        # print("top_predictions_cyc shape: ", top_predictions_cyc.shape)


        top_predictions = [top_predictions_car, top_predictions_ped, top_predictions_cyc, top_predictions_van]

        return top_predictions

           
           
def prepare_fusion_inputs(_3d_detector, example, _2d_stored_detections_path, num_classes, num_3d_raw_boxes, online_data):


    predictions2D = []

    if online_data is None:


        predictions2D = parse_offline_2D_detections(example, _2d_stored_detections_path)

    
    else:
        predictions2D = online_data['predictions2D']


    predictions3D = None


    if online_data is None:

        pred_3d_start_time = time.time()

        torch.cuda.synchronize()
        with torch.no_grad():
        
            output = _3d_detector(example, return_loss=False, pre_fusion=True)
            
        pred_3d_finish_time = time.time()

        print("time of 3d prediction in prepare_fusion_inputs: ", pred_3d_finish_time - pred_3d_start_time)

        predictions3D = output[0]

    else:
        predictions3D = online_data['predictions3D']
    
    #print("preds_dict: ", preds_dict)


    prepare_input_tensor_start_time = time.time()

    iou_test, tensor_index = prepare_input_tensor(example, predictions3D, predictions2D, num_classes, num_3d_raw_boxes, online_data)

    prepare_input_tensor_finish_time = time.time()

    print("time spent in prepare_input_tensor: ", prepare_input_tensor_finish_time - prepare_input_tensor_start_time)
 

    return predictions3D, iou_test, tensor_index




def prepare_input_tensor(example, predictions3D, predictions2D, num_classes, num_3d_raw_boxes, online_data):

    t1 = time.time()

    rect = None
    Trv2c = None
    P2 = None
    image_shape = None

    if online_data is None:

        rect = example["calib"]["rect"].squeeze().float()
        Trv2c = example["calib"]["Trv2c"].squeeze().float()
        P2 = example["calib"]["P2"].squeeze().float()

        image_shape = example["image"][0]["image_shape"]

    else:
        rect = online_data["calib"]["rect"].float()
        Trv2c = online_data["calib"]["Trv2c"].float()
        P2 = online_data["calib"]["P2"].float()

        image_shape = online_data["image_shape"]



    final_box_preds = predictions3D["box3d_lidar"].float()


    final_scores = predictions3D["scores"]


    final_box_preds_camera = box_torch_ops.box_lidar_to_camera(
        final_box_preds, rect, Trv2c)
    locs = final_box_preds_camera[:, :3]
    dims = final_box_preds_camera[:, 3:6]
    angles = final_box_preds_camera[:, 6]
    camera_box_origin = [0.5, 1.0, 0.5]
    box_corners = box_torch_ops.center_to_corner_box3d(
        locs, dims, angles, camera_box_origin, axis=1)

    box_corners_in_image = box_torch_ops.project_to_image(
        box_corners, P2)

    #print("box_corners_in_image tensor: ", box_corners_in_image)


    # box_corners_in_image: [N, 8, 2]
    minxy = torch.min(box_corners_in_image, dim=1)[0]
    maxxy = torch.max(box_corners_in_image, dim=1)[0]
    img_height = image_shape[0]
    img_width = image_shape[1]


    minxy[:,0] = torch.clamp(minxy[:,0],min = 0,max = img_width)
    minxy[:,1] = torch.clamp(minxy[:,1],min = 0,max = img_height)
    maxxy[:,0] = torch.clamp(maxxy[:,0],min = 0,max = img_width)
    maxxy[:,1] = torch.clamp(maxxy[:,1],min = 0,max = img_height)
    box_2d_preds = torch.cat([minxy, maxxy], dim=1)

    t2 = time.time()

    print("partial time 1 in prepare_input_tensor: ", t2-t1)

    t3 = time.time()


    dis_to_lidar = torch.norm(final_box_preds[:,:2],p=2,dim=1,keepdim=True)/82.0

    boxes_2d_detector = [np.zeros((np.maximum(1, predictions2D[i].shape[0]), 4)) for i in range(num_classes)]

    # boxes_2d_detector = [np.zeros((np.maximum(1, predictions2D[0].shape[0]), 4)),
    #                      np.zeros((np.maximum(1, predictions2D[1].shape[0]), 4)), 
    #                      np.zeros((np.maximum(1, predictions2D[2].shape[0]), 4)),
    #                      np.zeros((np.maximum(1, predictions2D[3].shape[0]), 4)),
    #                     ]

    boxes_2d_scores = [np.zeros((boxes_2d_detector[i].shape[0], 1)) for i in range(num_classes)]

    # boxes_2d_scores = [np.zeros((boxes_2d_detector[0].shape[0], 1)),
    #                    np.zeros((boxes_2d_detector[1].shape[0], 1)),
    #                    np.zeros((boxes_2d_detector[2].shape[0], 1)),
    #                    np.zeros((boxes_2d_detector[3].shape[0], 1)),
    #   
    
    for i in range(num_classes):
        boxes_2d_detector[i][0:predictions2D[i].shape[0],:]=predictions2D[i][0:predictions2D[i].shape[0],1:5]


    # boxes_2d_detector[0][0:predictions2D[0].shape[0],:]=predictions2D[0][0:predictions2D[0].shape[0],1:5]
    # boxes_2d_detector[1][0:predictions2D[1].shape[0],:]=predictions2D[1][0:predictions2D[1].shape[0],1:5]
    # boxes_2d_detector[2][0:predictions2D[2].shape[0],:]=predictions2D[2][0:predictions2D[2].shape[0],1:5]
    # boxes_2d_detector[3][0:predictions2D[3].shape[0],:]=predictions2D[3][0:predictions2D[3].shape[0],1:5]


    for i in range(num_classes):
        boxes_2d_scores[i][0:predictions2D[i].shape[0],:]=predictions2D[i][0:predictions2D[i].shape[0],5].reshape(-1,1)

    # boxes_2d_scores[0][0:predictions2D[0].shape[0],:]=predictions2D[0][0:predictions2D[0].shape[0],5].reshape(-1,1)
    # boxes_2d_scores[1][0:predictions2D[1].shape[0],:]=predictions2D[1][0:predictions2D[1].shape[0],5].reshape(-1,1)
    # boxes_2d_scores[2][0:predictions2D[2].shape[0],:]=predictions2D[2][0:predictions2D[2].shape[0],5].reshape(-1,1)
    # boxes_2d_scores[3][0:predictions2D[3].shape[0],:]=predictions2D[3][0:predictions2D[3].shape[0],5].reshape(-1,1)
  


    t4 = time.time()

    print("partial time 2 in prepare_input_tensor: ", t4-t3) 

    t5 = time.time()

    #time_gpu_to_cpu_start = time.time()
    box_2d_preds_numpy = box_2d_preds.detach().cpu().numpy()

    #print("box_2d_preds_numpy shape: ", box_2d_preds_numpy.shape)
    final_scores_numpy = final_scores.detach().cpu().numpy()
    #final_label_preds_numpy = final_label_preds.detach().cpu().numpy()
    dis_to_lidar_numpy = dis_to_lidar.detach().cpu().numpy()
    time_gpu_to_cpu_end = time.time()

    print("time of transfer from gpu to cpu: ", time_gpu_to_cpu_end - t5)

    overlaps = np.empty((num_classes,900000,4), dtype=np.float32)

    tensor_indices = np.empty((num_classes,900000,2), dtype=np.float32)


    non_empty_iou_test_tensor_list = []

    non_empty_tensor_index_tensor_list = []




    t6 = time.time()

    print("partial time 3 in prepare_input_tensor: ", t6-t5)  

    #box_2d_preds_numpy = box_2d_preds_numpy.astype(int)
    #boxes_2d_detector[0] = boxes_2d_detector[0].astype(int)
    #boxes_2d_detector[1] = boxes_2d_detector[1].astype(int)
    #boxes_2d_detector[2] = boxes_2d_detector[2].astype(int)

    if online_data is not None:
        final_scores_numpy = final_scores_numpy.astype(np.float32)



    # print("box_2d_preds_numpy dtype: ", box_2d_preds_numpy.dtype)
    # print("boxes_2d_detector[0] dtype: ", boxes_2d_detector[0].dtype)
    # print("final_scores_numpy dtype: ", final_scores_numpy.dtype)








    time_iou_build_start=time.time()

    # number = 53568
    # number = 74400


    for i in range(num_classes):
        iou_test,tensor_ind, max_num = eval.build_stage2_training(box_2d_preds_numpy[(i)*num_3d_raw_boxes:(i+1)*num_3d_raw_boxes, :],
                                            boxes_2d_detector[i],
                                            final_scores_numpy[(i)*num_3d_raw_boxes:(i+1)*num_3d_raw_boxes,:].reshape(-1,1),
                                            boxes_2d_scores[i],
                                            dis_to_lidar_numpy[(i)*num_3d_raw_boxes:(i+1)*num_3d_raw_boxes,:],
                                            overlaps[i],
                                            tensor_indices[i])


        iou_test_tensor = torch.FloatTensor(iou_test)
        iou_test_tensor = iou_test_tensor.permute(1,0)
        iou_test_tensor = iou_test_tensor.reshape(1,4,1,900000)

        tensor_ind = torch.LongTensor(tensor_ind)
        tensor_ind = tensor_ind.reshape(-1,2)

    
        if max_num == 0:
            non_empty_iou_test_tensor = torch.zeros(1,4,1,2)
            non_empty_iou_test_tensor[:,:,:,:] = -1
            non_empty_tensor_index_tensor = torch.zeros(2,2)
            non_empty_tensor_index_tensor[:,:] = -1
        else:
            non_empty_iou_test_tensor = iou_test_tensor[:,:,:,:max_num]
            non_empty_tensor_index_tensor = tensor_ind[:max_num,:]

        non_empty_iou_test_tensor_list.append(non_empty_iou_test_tensor)
        non_empty_tensor_index_tensor_list.append(non_empty_tensor_index_tensor)

    time_iou_build_end=time.time()

    print("time to build tensor: ", time_iou_build_end - time_iou_build_start)



    return non_empty_iou_test_tensor_list, non_empty_tensor_index_tensor_list
