_base_ = [
    './nus-3d.py',
    './default_runtime.py',
    './schedules/petr.py'
]
workflow = [('train', 1)]
plugin = True
plugin_dir = 'plugin/track/'

_samples_per_gpu = 1
_workers_per_gpu = 4

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False)

class_names = [
    'car', 'truck', 'bus', 'trailer',
    'motorcycle', 'bicycle', 'pedestrian',
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

model = dict(
    type='AlternatingDetAssoCamTracker',
    with_data_aug=True,
    feature_update_rate=None,
    use_grid_mask=True,  # use grid mask
    num_classes=7,
    num_query=500, # During pre-training of detctor was 900
    bbox_coder=dict(
        type='DETRTrack3DCoder',
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        max_num=100,
        num_classes=7),
    fix_feats=True,  # set fix feats to true can fix the backbone
    fix_neck=False,
    score_thresh=0.4,
    filter_score_thresh=0.35,
    miss_tolerance=5,
    affinity_thresh=0.3,
    hungarian=True,
    qim_args=dict(
        qim_type='QIMBase',
        merger_dropout=0, update_query_pos=True,
        fp_ratio=0.05, random_drop=0.1),  # hyper-param for query dropping mentioned in MOTR
    img_backbone=dict(
        type='VoVNetCP',
        spec_name='V-99-eSE',
        norm_eval=True,
        frozen_stages=-1,
        input_ch=3,
        out_features=('stage4','stage5',)),
    img_neck=dict(
        type='CPFPN',
        in_channels=[768, 1024],
        out_channels=256,
        num_outs=2), 
    loss_cfg=dict(
        type='ParallelMatcher',
        num_classes=7,
        weight_dict=None,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        assigner=dict(
            type='HungarianAssigner3DTrack',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            pc_range=point_cloud_range),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_asso=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=1.0,
            alpha=-1,
            loss_weight=10.0),
    ),
    pts_bbox_head=dict(
        type='PETRAssoTrackingHead',
        num_classes=7,
        in_channels=256,
        LID=True,
        with_position=True,
        with_multiview=True,
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        normedlinear=False,
        transformer=dict(
            type='PETRAssoTrackingTransformer',
            decoder=dict(
                type='PETRAssoTrackingTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                association_network=dict(
                    type='EdgeAugmentedTransformerAssociationLayer',
                    embed_dims=256,
                    ffn_dims=2048,
                    num_heads=8,
                    dropout=0.1,
                    norm_first=False,
                    cross_attn_value_gate=True,
                ),
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    attn_cfgs=[
                        # dict(
                        #     type='MaskedDetectionAndTrackSelfAttention',
                        #     embed_dims=256,
                        #     num_heads=8,
                        #     dropout=0.1,
                        #     block_det_to_track=False,
                        #     block_track_to_det=False),
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        # dict(
                        #     type='PETRMultiheadAttention',
                        #     embed_dims=256,
                        #     num_heads=8,
                        #     dropout=0.1),
                        dict(
                            type='PETRMultiheadFlashAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    with_cp=False,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        pc_range=point_cloud_range,
        positional_encoding=dict(
            type='SinePositionalEncoding3D',
            num_feats=128,
            normalize=True),),
    # model training and testing settings
    train_cfg=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        # dense_reg=1,
        # gaussian_overlap=0.1,
        # max_objs=500,
        # min_radius=2,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='GIoU3DCost', weight=0.0),
            pc_range=point_cloud_range)),
    test_cfg=dict(
        post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        pc_range=point_cloud_range[:2],
        max_per_img=500,
        max_pool_nms=False,
        min_radius=[4, 12, 10, 1, 0.85, 0.175],
        score_threshold=0.1,
        out_size_factor=4,
        voxel_size=voxel_size,
        nms_type='rotate',
        pre_max_size=1000,
        post_max_size=83,
        nms_thr=0.2))

dataset_type = 'NuScenesTrackDataset'
data_root = 'data/nuscenes/'

file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadMultiViewImageFromFilesCustom'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='InstanceRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Normalize3D', **img_norm_cfg),
    dict(type='Pad3D', size_divisor=32)]

train_pipeline_post = [
    dict(type='FormatBundle3DTrack'),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d',
         'instance_inds', 'img', 'timestamp', 'l2g_r_mat', 'l2g_t'],
         meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                    'depth2img', 'cam2img', 'pad_shape',
                    'scale_factor', 'flip', 'pcd_horizontal_flip',
                    'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                    'img_norm_cfg', 'pcd_trans', 'sample_idx',
                    'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                    'transformation_3d_flow', 'scene_token'))
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFilesCustom'),
    dict(type='Normalize3D', **img_norm_cfg),
    dict(type='Pad3D', size_divisor=32),
]

test_pipeline_post = [
    dict(type='FormatBundle3DTrack'),
    dict(type='Collect3D', keys=['img', 'timestamp', 'l2g_r_mat', 'l2g_t'],
         meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                    'depth2img', 'cam2img', 'pad_shape',
                    'scale_factor', 'flip', 'pcd_horizontal_flip',
                    'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                    'img_norm_cfg', 'pcd_trans', 'sample_idx',
                    'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                    'transformation_3d_flow', 'scene_token'))]


data = dict(
    samples_per_gpu=_samples_per_gpu,
    workers_per_gpu=_workers_per_gpu,
    train=dict(
        type=dataset_type,
        # number of frames for each clip in training. If you have more memory, I suggested you to use more.
        num_frames_per_sample=3,
        data_root=data_root,
        ann_file=data_root + 'ada_track_infos_train.pkl',
        pipeline_single=train_pipeline,
        pipeline_post=train_pipeline_post,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR'),
    # ),
    val=dict(type=dataset_type, pipeline_single=test_pipeline, pipeline_post=test_pipeline_post, classes=class_names, modality=input_modality,
             ann_file=data_root + 'ada_track_infos_val.pkl',
             data_root=data_root,
             num_frames_per_sample=1,),  # when inference, set bs=1
    test=dict(type=dataset_type, pipeline_single=test_pipeline,
              pipeline_post=test_pipeline_post,
              classes=class_names, modality=input_modality,
              ann_file=data_root + 'ada_track_infos_val.pkl',
              data_root=data_root,
              num_frames_per_sample=1,))  # when inference, set bs=1

# I suggest you to train longer. Like 48, 72 epochs, and change lr_config accrodingly
total_epochs = 24
evaluation = dict(interval=2)

runner = dict(type='EpochBasedRunner', max_epochs=24)

log_config = dict(
    tags={
        "samples_per_gpu": _samples_per_gpu,
        "workers_per_gpu": _workers_per_gpu,
        "total_epochs": total_epochs
    }
)


find_unused_parameters = True
# path to pretrained model.
load_from = "./work_dirs/pretrained/petr_vovnet_p4_1600_640_query500.pth"