_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/mot_challenge.py', '../_base_/default_runtime.py'
]

dataset_type='MOTChallengeDataset'
data_root='data/MOT17'
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),
    visualization=dict(type='TrackVisualizationHook', draw=False))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='TrackLocalVisualizer', vis_backends=vis_backends, name='visualizer')
# custom hooks
custom_hooks = [
    # Synchronize model buffers such as running_mean and running_var in BN
    # at the end of each epoch
    dict(type='SyncBuffersHook')
]

detector = _base_.model
detector.pop('data_preprocessor')
detector.rpn_head.bbox_coder.update(dict(clip_border=False))
detector.roi_head.bbox_head.update(dict(num_classes=1))
detector.roi_head.bbox_head.bbox_coder.update(dict(clip_border=False))
detector['init_cfg'] = dict(
    type='Pretrained',
    checkpoint=  # noqa: E251
    'https://download.openmmlab.com/mmtracking/mot/faster_rcnn/'
    'faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth')
del _base_.model

model = dict(
    type='DeepSORT',
    data_preprocessor=dict(
        type='TrackDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    detector=detector,
    reid=dict(
        type='BaseReID',
        data_preprocessor=dict(type='mmpretrain.ClsDataPreprocessor'),
        backbone=dict(
            type='mmpretrain.ResNet',
            depth=50,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch'),
        neck=dict(type='GlobalAveragePooling', kernel_size=(8, 4), stride=1),
        head=dict(
            type='LinearReIDHead',
            num_fcs=1,
            in_channels=2048,
            fc_channels=1024,
            out_channels=128,
            num_classes=380,
            loss_cls=dict(type='mmpretrain.CrossEntropyLoss', loss_weight=1.0),
            loss_triplet=dict(type='TripletLoss', margin=0.3, loss_weight=1.0),
            norm_cfg=dict(type='BN1d'),
            act_cfg=dict(type='ReLU')),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_iter25245-a452f51f.pth'  # noqa: E501
        )),
    tracker=dict(
        type='SORTTracker',
        motion=dict(type='KalmanFilter', center_only=False),
        obj_score_thr=0.5,
        reid=dict(
            num_samples=10,
            img_scale=(256, 128),
            img_norm_cfg=None,
            match_score_thr=2.0),
        match_iou_thr=0.5,
        momentums=None,
        num_tentatives=2,
        num_frames_retain=100))

train_dataloader = None

train_cfg = None
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

test_pipeline = [
    dict(
        type='TransformBroadcaster',
        transforms=[
            dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
            dict(type='Resize', scale=_base_.img_scale, keep_ratio=True),
            dict(
                type='Pad',
                size_divisor=32,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='LoadTrackAnnotations'),
        ]),
    dict(type='PackTrackInputs')
]
# test_dataloader = dict(
#     # Now StrongSORT only support video_based sampling
#     sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
#     dataset=dict(
#         _delete_=True,
#         # batch_size=1,  # 适当调整批量大小
#         # num_workers=2,  # 适当调整线程数
#         # persistent_workers=False,  # 避免在多线程环境下占用显存
#         type='MOTChallengeDataset',
#         data_root='data/MOT17',
#         ann_file='annotations/test_cocoformat.json',
#         data_prefix=dict(img_path='test'),
#         # when you evaluate track performance, you need to remove metainfo
#         test_mode=True,
#         pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    # Now we support two ways to test, image_based and video_based
    # if you want to use video_based sampling, you can use as follows
    # sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    sampler=dict(type='TrackImgSampler'),  # image-based sampling
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/test_cocoformat.json',
        data_prefix=dict(img_path='test'),
        test_mode=True,
        pipeline=test_pipeline))


# test_evaluator = dict(
#     type='MOTChallengeMetrics',
#     postprocess_tracklet_cfg=[
#         dict(type='InterpolateTracklets', min_num_frames=5, max_num_frames=20)
#     ],
#     format_only=True,s
#     outfile_prefix='./mot_17_test_res_deeps')