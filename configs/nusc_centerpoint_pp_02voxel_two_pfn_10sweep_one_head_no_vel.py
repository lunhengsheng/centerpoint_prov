import itertools
import logging
import numpy as np

from det3d.utils.config_tool import get_downsample_factor

tasks = [
    dict(num_class=5, class_names=["car", "trailer", "bus", "van", "pedestrian"]),
    # dict(num_class=1, class_names=["trailer"]),
    # dict(num_class=2, class_names=["bus", "van"]),
    # dict(num_class=1, class_names=["pedestrian"]),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

# training and testing settings
target_assigner = dict(
    tasks=tasks,
)

voxel_size = (0.2, 0.2, 8)
pc_range = (-70.4, -70.4, -10.0, 70.4, 70.4, 0.0)
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
        us_layer_strides=[0.5, 1, 2],
        us_num_filters=[128, 128, 128],
        num_input_features=64,
        logger=logging.getLogger("RPN"),
    ),
    bbox_head=dict(
        # type='RPNHead',
        type="CenterHead",
        in_channels=sum([128, 128, 128]),
        tasks=tasks,
        dataset='providentia',
        weight=0.25,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        common_heads={'reg': (2, 2), 'height': (1, 2), 'dim':(3, 2), 'rot':(2, 2)}, # (output_channel, num_conv)
    ),
)

assigner = dict(
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(model),
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
)


train_cfg = dict(assigner=assigner)

test_cfg = dict(
    post_center_limit_range=[-80, -40, -10.0, 80, 40, 0],
    max_per_img=500,
    nms=dict(
        nms_pre_max_size=1000,
        nms_post_max_size=83,
        nms_iou_threshold=0.2,
    ),
    score_threshold=0.1,
    pc_range=[-70.4, -70.4],
    out_size_factor=get_downsample_factor(model),
    voxel_size=[0.2, 0.2]
)

# dataset settings
dataset_type = "ProvidentiaDataset"
nsweeps = 1
data_root = "data3/providentia"

db_sampler = dict(
    type="GT-AUG",
    enable=False,
    db_info_path="data3/providentia/dbinfos_train_01sweeps_withvelo.pkl",
    sample_groups=[
        dict(car=4),
        dict(bus=2),
        dict(van=4),
        dict(trailer=6),
        dict(pedestrian=2)
    ], 
    db_prep_steps=[
        dict(
            filter_by_min_num_points=dict(
                car=5,
                bus=5,
                trailer=5,
                van=5,
                pedestrian=5,
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
    global_rot_noise=[-0.3925, 0.3925],
    global_scale_noise=[0.95, 1.05],
    db_sampler=db_sampler,
    class_names=class_names,
)

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
)

voxel_generator = dict(
    range=[-70.4, -70.4, -10.0, 70.4, 70.4, 0.0],
    voxel_size=[0.2, 0.2, 8],
    max_points_in_voxel=20,
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

train_anno = "data3/providentia/infos_train_01sweeps_gt.pkl"
val_anno = None #"data/nuScenes/infos_val_10sweeps_withvelo_filter_True.pkl"
test_anno = None

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        ann_file=train_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        test_mode=True,
        ann_file=val_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=test_anno,
        ann_file=test_anno,
        nsweeps=nsweeps,
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
    type="one_cycle", lr_max=0.001, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
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
total_epochs = 1200
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = './work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
load_from = None
resume_from = "models/nusc_one_head_no_vel/epoch_600.pth"
workflow = [('train', 1)]
