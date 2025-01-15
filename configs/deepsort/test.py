_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/mot_challenge.py', '../_base_/default_runtime.py'
]

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
pretrained='/home/kzy/project/PartDecoder/mmdetection/work_dirs/reid_pmnet_2xb12_mot17train100-mot17/best_reid-metric_mAP_iter_2000.pth'

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
    # reid=dict(
    #         type='BaseReID',
    #         data_preprocessor=dict(
    #             type='ReIDDataPreprocessor',
    #             mean=[123.675, 116.28, 103.53],
    #             std=[58.395, 57.12, 57.375],
    #             to_rgb=True),
    #         backbone=dict(
    #             type='mmpretrain.ResNet',
    #             depth=50,
    #             num_stages=4,
    #             out_indices=(1,2,3),
    #             style='pytorch',
    #             # init_cfg=dict(
    #             #     type='Pretrained',
    #             #     checkpoint=pretrained,
    #             #     prefix='backbone'
    #             # )
    #             ),
    #         neck=dict(
    #             type='PMNet',
    #                 num_queries=129,
    #                 embed_dims=256,
    #                 channel_mapper=dict( 
    #                     in_channels=[512,1024,2048],   # the output feature map dim
    #                     out_channels=256,
    #                     kernel_size=1,
    #                     norm_cfg=dict(type='BN'),
    #                     act_cfg=dict(type='LeakyReLU')
    #                     ),
    #                 encoder=dict(  
    #                     num_layers=6,
    #                     layer_cfg=dict(  
    #                         self_attn_cfg=dict(  # MultiheadAttention
    #                             embed_dims=256,
    #                             num_heads=8,
    #                             dropout=0.1,
    #                             batch_first=True),
    #                         ffn_cfg=dict(
    #                             embed_dims=256,
    #                             feedforward_channels=2048,
    #                             num_fcs=2,
    #                             ffn_drop=0.1,
    #                             act_cfg=dict(type='ReLU', inplace=True)))),
    #                 decoder=dict(
    #                     num_layers=6,
    #                     layer_cfg=dict(
    #                         self_attn_cfg=dict(
    #                             embed_dims=256,
    #                             num_heads=8,
    #                             attn_drop=0.1,
    #                             cross_attn=False),
    #                         cross_attn_cfg=dict(
    #                             embed_dims=256,
    #                             num_heads=8,
    #                             attn_drop=0.1,
    #                             cross_attn=True),
    #                         ffn_cfg=dict(
    #                             embed_dims=256,
    #                             feedforward_channels=2048,
    #                             num_fcs=2,
    #                             ffn_drop=0.1,
    #                             act_cfg=dict(type='ReLU', inplace=True)))
    #                 ),
    #                 positional_encoding=dict(num_feats=128, normalize=True),    # num_feats = len(x)+len(y)
    #             # init_cfg=dict(
    #             #     type='Pretrained',
    #             #     checkpoint=pretrained,
    #             #     prefix='neck'
    #             # ),
    #             ),
    #             head=dict(
    #                 type='LinearReIDHead',
    #                 num_fcs=1,
    #                 in_channels=256,
    #                 fc_channels=1024,
    #                 out_channels=256,
    #                 num_classes=380,
    #                 loss_cls=dict(type='mmpretrain.CrossEntropyLoss', loss_weight=1.0),
    #                 loss_triplet=dict(type='TripletLoss', margin=0.3, loss_weight=1.0),
    #                 norm_cfg=dict(type='BN1d'),
    #                 act_cfg=dict(type='ReLU')),
    #             init_cfg=dict(
    #                 type='Pretrained',
    #                 checkpoint=pretrained,
    #             )),   
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


train_pipeline = None
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

train_dataloader = None

train_cfg = None
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
