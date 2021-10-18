import os.path as osp
import numpy as np
import pickle
import random

from pathlib import Path
from functools import reduce
from typing import Tuple, List
import os 
import json 
from tqdm import tqdm
import argparse

from tqdm import tqdm


from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion



def get_obj(path):
    with open(path, 'rb') as f:
            obj = pickle.load(f)
    return obj 



import uuid 

class UUIDGeneration():
    def __init__(self):
        self.mapping = {}
    def get_uuid(self,seed):
        if seed not in self.mapping:
            self.mapping[seed] = uuid.uuid4().hex 
        return self.mapping[seed]
uuid_gen = UUIDGeneration()


def get_obj(path):
    with open(path) as f:
        obj = json.load(f)
    return obj


def _fill_infos(root_path, frames, split='train', nsweeps=1):
    # load all train infos
    infos = []
    for frame_name in tqdm(frames):  # global id
        lidar_path = os.path.join(root_path, split, frame_name[:5], 'pointclouds', frame_name[:-4]+"pcd")
        ref_path = os.path.join(root_path, split, frame_name[:5],'annotations', frame_name)

        ref_obj = get_obj(ref_path)
        
    #     ref_time = 1e-6 * int(ref_obj['frame_name'].split("_")[-1])

    #     ref_pose = np.reshape(ref_obj['veh_to_global'], [4, 4])
    #     _, ref_from_global = veh_pos_to_transform(ref_pose)

        info = {
            "lidar_path": lidar_path,
            "anno_path": ref_path, 
            "token": frame_name,
            "timestamp": "",
            "sweeps": []
        }

        if split != 'test':
            # read boxes 
            labels = ref_obj['labels']

            gt_boxes = []
            gt_names = []

            for label in labels:
                gt_box = []
                box = label["box3d"]

                location = [box["location"]["x"], 
                            box["location"]["y"],
                            box["location"]["z"] + 6.0]

                dimensions = [box["dimension"]["width"], 
                              box["dimension"]["length"], 
                              box["dimension"]["height"]]

                orientation = [box["orientation"]["rotationRoll"], 
                               box["orientation"]["rotationPitch"], 
                               box["orientation"]["rotationYaw"]]

                gt_box = location + dimensions + orientation

                gt_boxes.append(gt_box)

                gt_names.append(label["category"])
            
            gt_boxes = np.array(gt_boxes, dtype=np.float32)
            gt_names = np.array(gt_names, dtype=np.dtype('U10'))

            gt_boxes[:, -1] = -np.pi / 2 - gt_boxes[:, -1]

            
    #         if len(gt_boxes) != 0:
    #             # transform from Waymo to KITTI coordinate 
    #             # Waymo: x, y, z, length, width, height, rotation from positive x axis clockwisely
    #             # KITTI: x, y, z, width, length, height, rotation from negative y axis counterclockwisely 
    #             gt_boxes[:, -1] = -np.pi / 2 - gt_boxes[:, -1]
    #             gt_boxes[:, [3, 4]] = gt_boxes[:, [4, 3]]

    #         gt_names = np.array([TYPE_LIST[ann['label']] for ann in annos])
    #         mask_not_zero = (num_points_in_gt > 0).reshape(-1)    

    #         # filter boxes without lidar points 
            info['gt_boxes'] = gt_boxes
            info['gt_names'] = gt_names

        infos.append(info)
    return infos


def sort_frame(frames):
    indices = [] 

    for f in frames:
        seq_id = int(f.split("_")[1])
        frame_id= int(f.split("_")[2][:-5])

        idx = seq_id * 1000 + frame_id
        indices.append(idx)

    rank = list(np.argsort(np.array(indices)))

    frames = [frames[r] for r in rank]
    return frames

def get_available_frames(root, split):
    dir_path = os.path.join(root, split)
    available_sequences = [os.path.join(dir_path, seq) for seq in os.listdir(dir_path)]

    available_frames = []

    for seq in available_sequences:
        seq = os.path.join(seq, "annotations")
        frames_in_sequence = list(os.listdir(seq))
        available_frames += frames_in_sequence

        

    sorted_frames = sort_frame(available_frames)

    print(split, " split ", "exist frame num:", len(sorted_frames))
    return sorted_frames





def create_providentia_infos(root_path, split='train', nsweeps=1):
    frames = get_available_frames(root_path, split)

    providentia_infos = _fill_infos(
        root_path, frames, split, nsweeps
    )

    print(
        f"sample: {len(providentia_infos)}"
    )
    with open(
        os.path.join(root_path, "infos_"+split+"_{:02d}sweeps_gt.pkl".format(nsweeps)), "wb"
    ) as f:
        pickle.dump(providentia_infos, f)


def parse_args():
    parser = argparse.ArgumentParser(description="Providentia 3D Extractor")
    parser.add_argument("--path", type=str, default="data3/Providentia/train")
    parser.add_argument("--info_path", type=str)
    parser.add_argument("--result_path", type=str)
    parser.add_argument("--gt", action='store_true' )
    parser.add_argument("--tracking", action='store_true')
    args = parser.parse_args()
    return args


def reorganize_info(infos):
    new_info = {}

    for info in infos:
        token = info['token']
        new_info[token] = info

    return new_info 

if __name__ == "__main__":
    args = parse_args()

    with open(args.info_path, 'rb') as f:
        infos = pickle.load(f)
    
    if args.gt:
        _create_gt_detection(infos, tracking=args.tracking)
        exit() 

    infos = reorganize_info(infos)
    with open(args.path, 'rb') as f:
        preds = pickle.load(f)
    _create_pd_detection(preds, infos, args.result_path, tracking=args.tracking)
