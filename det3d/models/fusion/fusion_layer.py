from ..registry import FUSION

from .fusion_utils import prepare_fusion_inputs

from collections import defaultdict




from det3d.models import build_detector
from det3d.torchie import Config
from det3d.core import box_torch_ops

from det3d.datasets.utils import eval
from det3d.models.losses.focal_loss import SigmoidFocalClassificationLoss

from det3d.models.losses.centernet_loss import FastFocalLoss, RegLoss

import time
import logging
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from det3d.models.utils import Sequential

import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')


class CLOCsFusion(nn.Module):
    def __init__(self):
        super(CLOCsFusion, self).__init__()
        self.fuse_2d_3d = Sequential(
            nn.Conv2d(4,18,1),
            nn.ReLU(),
            nn.Conv2d(18,36,1),
            nn.ReLU(),
            nn.Conv2d(36,36,1),
            nn.ReLU(),
            nn.Conv2d(36,1,1),
        )

        self.fuse_2d_3d = self.fuse_2d_3d.cuda()

    def forward(self, input):
        out = self.fuse_2d_3d(input)
        return out







@FUSION.register_module
class FusionLayer(nn.Module):
    def __init__(self, name, num_classes, _3d_net_cfg_path, _3d_net_path, _2d_data_path, _3d_raw_boxes_shape):
        super(FusionLayer, self).__init__()
        self.name = name

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self._3d_detector_config_path = _3d_net_cfg_path

        self._2d_stored_detections_path = _2d_data_path

        self._3d_net_cfg = Config.fromfile(self._3d_detector_config_path)
        self._3d_detector = build_detector(self._3d_net_cfg.model, train_cfg=None, test_cfg=self._3d_net_cfg.test_cfg)
        checkpoint = torch.load(_3d_net_path)


        self._3d_detector.load_state_dict(checkpoint["state_dict"])

    


        self._3d_detector = self._3d_detector.to(self.device).eval().freeze()


        self.bbox_head = self._3d_detector.bbox_head

        self.num_classes = num_classes

        self._3d_raw_boxes_H = _3d_raw_boxes_shape[0]
        self._3d_raw_boxes_W = _3d_raw_boxes_shape[1]

        self.num_3d_raw_boxes = self._3d_raw_boxes_H * self._3d_raw_boxes_W

        self.crit = FastFocalLoss()
        self.crit_reg = RegLoss()


        logger = logging.getLogger("Fusion Layer")
        self.logger = logger


        self.focal_loss = SigmoidFocalClassificationLoss()
    
        self.corner_points_feature = Sequential(
            nn.Conv2d(24,48,1),
            nn.ReLU(),
            nn.Conv2d(48,96,1),
            nn.ReLU(),
            nn.Conv2d(96,96,1),
            nn.ReLU(),
            nn.Conv2d(96,4,1),
        )

        self.maxpool = Sequential(
            nn.MaxPool2d([100,1],1),
        )

        self.tasks = nn.ModuleList()

        for j in range(self.num_classes):
            self.tasks.append(CLOCsFusion())



    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info("load model from: {}".format(pretrained))



    
    def fusion(self, input_1_list, tensor_index_list, online_data):

        flag = -1

        x_list = []

        for i in range(len(input_1_list)):

            tensor_index_list[i] = tensor_index_list[i].cuda()

            input_1_list[i] = input_1_list[i].cuda()

            if online_data is not None:
                input_1_list[i] = input_1_list[i].cuda().half()

            if tensor_index_list[i][0,0] == -1:
                out_1 = torch.zeros(1,100,self.num_3d_raw_boxes,dtype = input_1_list[i].dtype,device = input_1_list[i].device)
                out_1[:,:,:] = -9999
                flag = 0
            else:
                x = self.tasks[i](input_1_list[i])
                out_1 = torch.zeros(1,100,self.num_3d_raw_boxes,dtype = input_1_list[i].dtype,device = input_1_list[i].device)
                out_1[:,:,:] = -9999
                out_1[:,tensor_index_list[i][:,0],tensor_index_list[i][:,1]] = x[0,:,0,:]
                flag = 1
            x = self.maxpool(out_1)
            #x, _ = torch.max(out_1,1)
            x = x.squeeze().reshape(1,-1,1)
            print("x shape in fusion: ", x.shape)
            #compact_tensor[0:,0:,i]=x[0:,0:,0]
            x_list.append(x)
        
        return x_list,flag

    def _sigmoid(self, x):
        y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
        return y


    def loss(self, example, fused_hm_preds, flag, **kwargs):   

        rets = []
 

        for task_id in range(self.num_classes):


            fused_hm_preds[task_id] = self._sigmoid(fused_hm_preds[task_id])


            hm_loss = self.crit(fused_hm_preds[task_id], example['hm'][task_id], example['ind'][task_id], example['mask'][task_id], example['cat'][task_id])


            ret = {}


            ret.update({'loss': hm_loss})

            rets.append(ret)


        """convert batch-key to key-batch
        """
        rets_merged = defaultdict(list)
        for ret in rets:
            for k, v in ret.items():
                rets_merged[k].append(v)

        return rets_merged

        
    @torch.no_grad()
    def predict(self, example, test_cfg, box_preds, fused_hm):
        """decode, nms, then return the detection result
        """
        # get loss info
        rets = []
        metas = []

        post_center_range = test_cfg.post_center_limit_range
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=fused_hm[0].dtype,
                device=fused_hm[0].device,
            )

        for task_id in range(self.num_classes):
            batch_size = fused_hm[task_id].shape[0]
            
            if example is not None:

                if "metadata" not in example or len(example["metadata"]) == 0:
                    meta_list = [None] * batch_size
                else:
                    meta_list = example["metadata"]

                metas.append(meta_list)

            batch_hm = torch.sigmoid(fused_hm[task_id])


            batch_box_preds = box_preds[task_id*self.num_3d_raw_boxes:(task_id+1)*self.num_3d_raw_boxes,:].unsqueeze(0)
            rets.append(self.post_processing(batch_box_preds, batch_hm, test_cfg, post_center_range)) 


        # Merge branches results
        ret_list = []
        num_samples = len(rets[0])

        for i in range(num_samples):
            ret = {}
            for k in rets[0][i].keys():
                if k in ["box3d_lidar", "scores"]:
                    ret[k] = torch.cat([ret[i][k] for ret in rets])
                elif k in ["label_preds"]:
                    flag = 0
                    for j in range(self.num_classes):
                        rets[j][i][k] += flag
                        flag += 1
                    ret[k] = torch.cat([ret[i][k] for ret in rets])
            
            if example is not None:
                ret['metadata'] = metas[0][i]

            ret_list.append(ret)

        return ret_list 


    @torch.no_grad()
    def post_processing(self, batch_box_preds, batch_hm, test_cfg, post_center_range):
        batch_size = len(batch_hm)

        prediction_dicts = []
        for i in range(batch_size):
            box_preds = batch_box_preds[i]
            hm_preds = batch_hm[i]

            scores, labels = torch.max(hm_preds, dim=-1)


            print('scores shape before mask: ', scores.shape)
  

            score_mask = scores > test_cfg.score_threshold
            distance_mask = (box_preds[..., :3] >= post_center_range[:3]).all(1) \
                & (box_preds[..., :3] <= post_center_range[3:]).all(1)
        
            mask = distance_mask & score_mask 

            box_preds = box_preds[mask]
            scores = scores[mask]
            labels = labels[mask]

            print('scores shape after mask: ', scores.shape)


            boxes_for_nms = box_preds[:, [0, 1, 2, 3, 4, 5, -1]]

            selected = box_torch_ops.rotate_nms_pcdet(boxes_for_nms.float(), scores.float(), 
                                thresh=test_cfg.nms.nms_iou_threshold,
                                pre_maxsize=test_cfg.nms.nms_pre_max_size,
                                post_max_size=test_cfg.nms.nms_post_max_size)

            selected_boxes = box_preds[selected]
            selected_scores = scores[selected]
            selected_labels = labels[selected]

            prediction_dict = {
                'box3d_lidar': selected_boxes,
                'scores': selected_scores,
                'label_preds': selected_labels
            }

            prediction_dicts.append(prediction_dict)

        return prediction_dicts 



    def forward(self, example, return_loss=True, online_data = None):


        predictions3D, fusion_input_list, tensor_index_list = prepare_fusion_inputs(self._3d_detector, example, self._2d_stored_detections_path, self.num_classes, self.num_3d_raw_boxes, online_data)
        

        time_before_fusion = time.time()


        fused_hm_preds, flag = self.fusion(fusion_input_list,tensor_index_list, online_data)


        time_after_fusion = time.time()

        print("time of clocs inference: ", time_after_fusion - time_before_fusion)



        if return_loss:

            for i in range(self.num_classes):

                #fused_hm_preds[i] = fused_hm_preds[i].reshape(1,248,216,1)
                fused_hm_preds[i] = fused_hm_preds[i].reshape(1, self._3d_raw_boxes_H, self._3d_raw_boxes_W,1)

                fused_hm_preds[i] = fused_hm_preds[i].permute(0, 3, 1, 2).contiguous()
    

            return self.loss(example, fused_hm_preds, flag)

        else:


            time_predict_start = time.time()

            results = self.predict(example, self._3d_detector.test_cfg, predictions3D["box3d_lidar"], fused_hm_preds)

            time_predict_finish = time.time()

            print("time of prediction in evaluation: ", time_predict_finish - time_predict_start)

            return results
 

  
