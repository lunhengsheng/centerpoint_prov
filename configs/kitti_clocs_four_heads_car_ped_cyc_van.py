import itertools
import logging
import numpy as np

from det3d.utils.config_tool import get_downsample_factor




tasks = [
    dict(num_class=1, class_names=["Car"]),
    dict(num_class=1, class_names=["Pedestrian"]),
    dict(num_class=1, class_names=["Cyclist"]),
    dict(num_class=1, class_names=["Van"])
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

# training and testing settings
target_assigner = dict(
    tasks=tasks,
)

pc_range = [0, -39.68, -4, 96, 39.68, 4]
voxel_size = [0.16, 0.16, 8.0]
grid_size = (np.asarray(pc_range)[3:] - np.asarray(pc_range)[:3])/ np.asarray(voxel_size)


# model settings
model = dict(
    type="PointPillars",
    pretrained=None,
    reader=dict(
        type="PillarFeatureNet",
        num_filters=[64, 64],
        num_input_features=4,
        with_distance=False,
        voxel_size=voxel_size,
        pc_range=pc_range,
    ),
    backbone=dict(type="PointPillarsScatter", ds_factor=1),
    neck=dict(
        type="RPN",
        layer_nums=[3, 5, 5],
        ds_layer_strides=[2, 2, 2],
        ds_num_filters=[64, 128, 256],
        us_layer_strides=[1, 2, 4],
        us_num_filters=[128, 128, 128],
        num_input_features=64,
        logger=logging.getLogger("RPN"),
    ),
    bbox_head=dict(
        # type='RPNHead',
        type="CenterHead",
        in_channels=sum([128, 128, 128]),
        tasks=tasks,
        dataset='kitti',
        weight=1.5,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        common_heads={'reg': (2, 2), 'height': (1, 2), 'dim':(3, 2), 'rot':(2, 2)}, # (output_channel, num_conv)
    ),
)





# fusion settings
fusion = dict(
    type="FusionLayer",
    name = "clocs",
    num_classes = 4,
    _3d_net_cfg_path = 'configs/kitti_centerpoint_four_heads_car_ped_cyc_van.py',
     _3d_net_path = 'models/kitti_centerpoint_four_heads_car_ped_cyc_van/epoch_26.pth',
     _2d_data_path = '2D_predictions_yolov4_kitti',
    _3d_raw_boxes_shape = [248,300]

)

assigner = dict(
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(model),
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
)


assigner = dict(
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(model),
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
)


train_cfg = dict(assigner=assigner)

# test_cfg = dict(
#     post_center_limit_range=[0, -40, -4.0, 70, 40, 2],
#     max_per_img=500,
#     nms=dict(
#         nms_pre_max_size=1000,
#         nms_post_max_size=83,
#         nms_iou_threshold=0.2,
#     ),
#     score_threshold=0.1,
#     pc_range=[0, -39.68],
#     out_size_factor=get_downsample_factor(model),
#     voxel_size=[0.16, 0.16]
# )

# dataset settings
dataset_type = "KittiDataset"
data_root = "data/kitti"

db_sampler = dict(
    type="GT-AUG",
    enable=True,
    db_info_path="data/kitti/centerpoint_pkl/dbinfos_train.pkl",
    sample_groups=[
        #dict(Car=15),
        #dict(Pedestrian=8),
        #dict(Cyclist=10),
        #dict(Van=10),
    ], 
    db_prep_steps=[
        dict(
            filter_by_min_num_points=dict(
                Car=5,
                Pedestrian=5,
                Cyclist=5,
                Van = 5,
            )
        ),
        dict(filter_by_difficulty=[-1],),
    ],
    global_random_rotation_range_per_object=[0, 0],
    rate=1.0,
)

train_preprocessor = dict(
    mode="train",
    shuffle_points=True,
    gt_loc_noise=[0.25, 0.25, 0.25],
    gt_rot_noise=[-0.15707963267, 0.15707963267],
    global_rot_noise=[-0.78539816, 0.78539816],
    global_scale_noise=[0.95, 1.05],
    global_rot_per_obj_range=[0, 0],
    global_trans_noise=[0.0, 0.0, 0.0],
    remove_points_after_sample=True,
    gt_drop_percentage=0.0,
    gt_drop_max_keep_points=15,
    remove_unknown_examples=False,
    remove_environment=False,
    db_sampler=db_sampler,
    class_names=class_names,
)

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
    remove_environment=False,
    remove_unknown_examples=False,
)

voxel_generator = dict(
    range = [0, -39.68, -4, 96, 39.68, 4],
    voxel_size = [0.16, 0.16, 8.0],
    max_points_in_voxel=100,
    max_voxel_num=[30000, 60000],
)

train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=train_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]
test_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=val_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]

train_anno = "/data/kitti/centerpoint_pkl/kitti_infos_train.pkl"
val_anno = "/data/kitti/centerpoint_pkl/kitti_infos_val.pkl"
test_anno = None

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=6,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=data_root + "/centerpoint_pkl/kitti_infos_train.pkl",
        ann_file=train_anno,
        class_names=class_names,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=data_root + "/centerpoint_pkl/kitti_infos_val.pkl",
        ann_file=val_anno,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=test_anno,
        ann_file=test_anno,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
)


optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# optimizer
optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
)
lr_config = dict(
    type="one_cycle", lr_max=0.003, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
)

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type='TensorboardLoggerHook')
    ],
)
# yapf:enable
# runtime settings
total_epochs = 25
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = './work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
load_from = None
resume_from = None #'models/kitti_clocs_only_car_ped_cyc/epoch_5.pth'
workflow = [("train",1),("val", 1)]
