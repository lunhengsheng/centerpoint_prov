"""
PointPillars fork from SECOND.
Code written by Alex Lang and Oscar Beijbom, 2018.
Licensed under MIT License [see LICENSE].
"""

import torch
from det3d.models.utils import get_paddings_indicator
from torch import nn
from torch.nn import functional as F
from ..registry import BACKBONES, READERS
from ..utils import build_norm_layer
#from det3d.ops.voxel import DynamicScatter

class PFNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None, last_layer=False):
        """
        Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """

        super().__init__()
        self.name = "PFNLayer"
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)
        self.norm_cfg = norm_cfg

        self.linear = nn.Linear(in_channels, self.units, bias=False)
        self.norm = build_norm_layer(self.norm_cfg, self.units)[1]

    def forward(self, inputs):

        x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        torch.backends.cudnn.enabled = True
        x = F.relu(x)

        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


@READERS.register_module
class PillarFeatureNet(nn.Module):
    def __init__(
        self,
        num_input_features=4,
        num_filters=(64,),
        with_distance=False,
        voxel_size=(0.2, 0.2, 4),
        pc_range=(0, -40, -3, 70.4, 40, 1),
        norm_cfg=None,
    ):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """

        super().__init__()
        self.name = "PillarFeatureNet"
        assert len(num_filters) > 0

        self.num_input_features = num_input_features
        self.num_input_features += 5
        if with_distance:
            self.num_input_features += 1
        self._with_distance = with_distance

        # Create PillarFeatureNet layers
        num_filters = [self.num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(
                    in_filters, out_filters, norm_cfg=norm_cfg, last_layer=last_layer
                )
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.pc_range = pc_range
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]

    def forward(self, features, num_voxels, coors):
        device = features.device

        dtype = features.dtype

        # Find distance of x, y, and z from cluster center
        # features = features[:, :, :self.num_input]
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(
            features
        ).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center
        # f_center = features[:, :, :2]
        f_center = torch.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (
            coors[:, 3].to(dtype).unsqueeze(1) * self.vx + self.x_offset
        )
        f_center[:, :, 1] = features[:, :, 1] - (
            coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset
        )

        # Combine together feature decorations
        features_ls = [features, f_cluster, f_center]
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        batch_dict = dict()

        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            features = pfn(features)

        batch_dict["pillar_features"] = features.squeeze()

        return batch_dict

@READERS.register_module
class DynamicPillarFeatureNet(PillarFeatureNet):
    """Pillar Feature Net using dynamic voxelization.
    The network prepares the pillar features and performs forward pass
    through PFNLayers. The main difference is that it is used for
    dynamic voxels, which contains different number of points inside a voxel
    without limits.
    Args:
        num_input_features (int, optional): Number of input features,
            either x, y, z or x, y, z, r. Defaults to 4.
        num_filters (tuple, optional): Number of features in each of the
            N PFNLayers. Defaults to (64, ).
        with_distance (bool, optional): Whether to include Euclidean distance
            to points. Defaults to False.
        with_cluster_center (bool, optional): [description]. Defaults to True.
        with_voxel_center (bool, optional): [description]. Defaults to True.
        voxel_size (tuple[float], optional): Size of voxels, only utilize x
            and y size. Defaults to (0.2, 0.2, 4).
        pc_range (tuple[float], optional): Point cloud range, only
            utilizes x and y min. Defaults to (0, -40, -3, 70.4, 40, 1).
        norm_cfg ([type], optional): [description].
            Defaults to dict(type='BN1d', eps=1e-3, momentum=0.01).
        mode (str, optional): The mode to gather point features. Options are
            'max' or 'avg'. Defaults to 'max'.
    """

    def __init__(self,
                 num_input_features=4,
                 num_filters=(64, ),
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 ):
        super(DynamicPillarFeatureNet, self).__init__(
            num_input_features,
            num_filters,
            with_distance,
            voxel_size=voxel_size,
            pc_range=pc_range,
            norm_cfg=norm_cfg,
            )
        self.fp16_enabled = False
        num_filters = [self.num_input_features] + list(num_filters)
        pfn_layers = []
        # TODO: currently only support one PFNLayer

        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i > 0:
                in_filters *= 2
            norm_name, norm_layer = build_norm_layer(norm_cfg, out_filters)
            pfn_layers.append(
                nn.Sequential(
                    nn.Linear(in_filters, out_filters, bias=False), norm_layer,
                    nn.ReLU(inplace=True)))
        self.num_pfn = len(pfn_layers)
        self.pfn_layers = nn.ModuleList(pfn_layers)
        self.pfn_scatter = DynamicScatter(voxel_size, pc_range,
                                         average_points=False)
        self.cluster_scatter = DynamicScatter(
            voxel_size, pc_range, average_points=True)

    def map_voxel_center_to_point(self, pts_coors, voxel_mean, voxel_coors):
        """Map the centers of voxels to its corresponding points.
        Args:
            pts_coors (torch.Tensor): The coordinates of each points, shape
                (M, 3), where M is the number of points.
            voxel_mean (torch.Tensor): The mean or aggreagated features of a
                voxel, shape (N, C), where N is the number of voxels.
            voxel_coors (torch.Tensor): The coordinates of each voxel.
        Returns:
            torch.Tensor: Corresponding voxel centers of each points, shape
                (M, C), where M is the numver of points.
        """
        # Step 1: scatter voxel into canvas
        # Calculate necessary things for canvas creation
        canvas_y = int(
            (self.pc_range[4] - self.pc_range[1]) / self.vy)
        canvas_x = int(
            (self.pc_range[3] - self.pc_range[0]) / self.vx)
        canvas_channel = voxel_mean.size(1)
        batch_size = pts_coors[-1, 0] + 1
        canvas_len = canvas_y * canvas_x * batch_size
        # Create the canvas for this sample
        canvas = voxel_mean.new_zeros(canvas_channel, canvas_len)
        # Only include non-empty pillars
        indices = (
            voxel_coors[:, 0] * canvas_y * canvas_x +
            voxel_coors[:, 2] * canvas_x + voxel_coors[:, 3])
        # Scatter the blob back to the canvas
        canvas[:, indices.long()] = voxel_mean.t()

        # Step 2: get voxel mean for each point
        voxel_index = (
            pts_coors[:, 0] * canvas_y * canvas_x +
            pts_coors[:, 2] * canvas_x + pts_coors[:, 3])
        center_per_point = canvas[:, voxel_index.long()].t()
        return center_per_point

    def forward(self, features, coors):
        """Forward function.
        Args:
            features (torch.Tensor): Point features or raw points in shape
                (N, M, C).
            coors (torch.Tensor): Coordinates of each voxel
        Returns:
            torch.Tensor: Features of pillars.
        """
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        
        voxel_mean, mean_coors = self.cluster_scatter(features, coors)
        points_mean = self.map_voxel_center_to_point(
            coors, voxel_mean, mean_coors)
        # TODO: maybe also do cluster for reflectivity
        f_cluster = features[:, :3] - points_mean[:, :3]
        features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
    
        f_center = features.new_zeros(size=(features.size(0), 2))
        f_center[:, 0] = features[:, 0] - (
            coors[:, 3].type_as(features) * self.vx + self.x_offset)
        f_center[:, 1] = features[:, 1] - (
            coors[:, 2].type_as(features) * self.vy + self.y_offset)
        features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :3], 2, 1, keepdim=True)
            features_ls.append(points_dist)
        
        batch_dict = dict()

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)
        for i, pfn in enumerate(self.pfn_layers):
            point_feats = pfn(features)
            voxel_feats, voxel_coors = self.pfn_scatter(point_feats, coors)
            if i != len(self.pfn_layers) - 1:
                # need to concat voxel feats if it is not the last pfn
                feat_per_point = self.map_voxel_center_to_point(
                    coors, voxel_feats, voxel_coors)
                features = torch.cat([point_feats, feat_per_point], dim=1)

        batch_dict["pillar_features"] = voxel_feats
        batch_dict["feature_coors"] = voxel_coors
        return batch_dict

@BACKBONES.register_module
class PointPillarsScatter(nn.Module):
    def __init__(
        self, num_input_features=64, norm_cfg=None, name="PointPillarsScatter", **kwargs
    ):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.name = "PointPillarsScatter"
        self.nchannels = num_input_features

    def forward(self, batch_dict, coords, batch_size, input_shape):

        self.nx = input_shape[0]
        self.ny = input_shape[1]

        voxel_features = batch_dict["pillar_features"]
        batch_dict["voxel_coords"] = coords

        batch_dict["batch_size"] = batch_size

        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                self.nchannels,
                self.nx * self.ny,
                dtype=voxel_features.dtype,
                device=voxel_features.device,
            )

            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_itt

            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, self.nchannels, self.ny, self.nx)
        batch_dict["spatial_features"] = batch_canvas

        return batch_dict
