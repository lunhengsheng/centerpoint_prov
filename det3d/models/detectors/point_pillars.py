from numpy.testing._private.utils import KnownFailureException
from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from copy import deepcopy 
#from det3d.ops.voxel import Voxelization
import torch
from torch.nn import functional as F
import time



import copy


@DETECTORS.register_module
class PointPillars(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        encoder2d = None,
        neck = None,
        cfe = None,
        decoder2d = None,     
        bbox_head = None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        dyn_voxelizer=None
    ):
        super(PointPillars, self).__init__(
            reader, backbone, encoder2d, neck, cfe, decoder2d, bbox_head, train_cfg, test_cfg, pretrained
        )

        self.dyn_voxelizer=None

        if(dyn_voxelizer is not None):
            self.dyn_voxelizer = Voxelization(**dyn_voxelizer) 

    def extract_feat(self, data):
        batch_dict = self.reader(
            data["features"], data["num_voxels"], data["coors"]
        )

        if self.with_backbone:
            batch_dict = self.backbone(
                batch_dict, data["coors"], data["batch_size"], data["input_shape"]
            )
        if self.with_neck:
            batch_dict = self.neck(batch_dict)

        elif self.with_encoder2d:
            batch_dict = self.encoder2d(batch_dict)

            batch_dict = self.cfe(batch_dict)
            batch_dict = self.decoder2d(batch_dict)
        
        return batch_dict

    @torch.no_grad()
    def voxelize(self, points, batch_size):
        """Apply dynamic voxelization to points.
        Args:
            points (list[torch.Tensor]): Points of each sample.
        Returns:
            tuple[torch.Tensor]: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provide a coors mapping

        points_batch = []
        print("points shape is: ", points.shape)

        for i in range(batch_size):
            mask = i==points[:,0]
            temp = points[mask][:,1:]

            print("temp shape is: ", temp.shape)
            points_batch.append(temp.contiguous())

        for res in points_batch:
            res_coors = self.dyn_voxelizer(res)
            coors.append(res_coors)
        points = torch.cat(points_batch, dim=0)
        coors_batch = []

        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch


    def extract_feat_dyn(self, data):
        """Extract features from points."""
        voxels, coors = self.voxelize(data["points"], data["batch_size"])
        batch_dict = self.reader(voxels, coors)
        batch_dict = self.backbone(batch_dict, batch_dict["feature_coors"], data["batch_size"], data["input_shape"])
        if self.with_neck:
            batch_dict = self.neck(batch_dict)
        return batch_dict

    def forward(self, example, return_loss=True, **kwargs):

        #print("example keys: ", example.keys())
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]
        points = example["points"]

        batch_size = len(num_voxels)
        print("batch_size is :", batch_size)

        data = dict(
            points=points,
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        if self.dyn_voxelizer is not None:
            batch_dict = self.extract_feat_dyn(data)
        else:
            batch_dict = self.extract_feat(data)

        preds = self.bbox_head(batch_dict)

        


        if return_loss:

            return self.bbox_head.loss(example, preds)
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg, **kwargs)

            
    def forward_two_stage(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        batch_dict = self.extract_feat(data)
        bev_feature = batch_dict["spatial_features"] 
        preds = self.bbox_head(batch_dict)

        # manual deepcopy ...
        new_preds = []
        for pred in preds:
            new_pred = {} 
            for k, v in pred.items():
                new_pred[k] = v.detach()

            new_preds.append(new_pred)

        boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

        if return_loss:
            return boxes, bev_feature, self.bbox_head.loss(example, preds)
        else:
            return boxes, bev_feature, None 
