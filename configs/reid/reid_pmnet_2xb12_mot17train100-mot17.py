_base_ = [
    '../_base_/datasets/mot_challenge_reid.py', '../_base_/default_runtime.py'
]
pretrained='/home/kzy/project/PartDecoder/mmdetection/work_dirs/reid_pmnet_2xb32_mot17train80_test-mot17val20/best_reid-metric_mAP_iter_3000.pth'
model = dict(
    type='BaseReID',
    data_preprocessor=dict(
        type='ReIDDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    backbone=dict(
        type='mmpretrain.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1,2,3),
        style='pytorch',
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint=pretrained,
        #     prefix='backbone'
        # )
        ),
     neck=dict(
        type='PMNet',
            num_queries=129,
            embed_dims=256,
            channel_mapper=dict( 
                in_channels=[512,1024,2048],   # the output feature map dim
                out_channels=256,
                kernel_size=1,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='LeakyReLU')
                ),
            encoder=dict(  
                num_layers=6,
                layer_cfg=dict(  
                    self_attn_cfg=dict(  # MultiheadAttention
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.1,
                        batch_first=True),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=2048,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type='ReLU', inplace=True)))),
            decoder=dict(
                num_layers=6,
                layer_cfg=dict(
                    self_attn_cfg=dict(
                        embed_dims=256,
                        num_heads=8,
                        attn_drop=0.1,
                        cross_attn=False),
                    cross_attn_cfg=dict(
                        embed_dims=256,
                        num_heads=8,
                        attn_drop=0.1,
                        cross_attn=True),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=2048,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type='ReLU', inplace=True)))
            ),
            positional_encoding=dict(num_feats=128, normalize=True),    # num_feats = len(x)+len(y)
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint=pretrained,
        #     prefix='neck'
        # ),
    ),
    head=dict(
        type='LinearReIDHead',
        num_fcs=1,
        in_channels=256,
        fc_channels=1024,
        out_channels=256,
        num_classes=380,            # 477
        loss_cls=dict(type='mmpretrain.CrossEntropyLoss', loss_weight=1.0),
        loss_triplet=dict(type='TripletLoss', margin=0.3, loss_weight=1.0),
        norm_cfg=dict(type='BN1d'),
        act_cfg=dict(type='ReLU')),
    init_cfg=dict(
        type='Pretrained',
        checkpoint=pretrained,
    )
    )

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    clip_grad=None,
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 1000,
        by_epoch=False,          # IterBased 修改 !
        begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=6,
        # by_epoch=True,
        by_epoch=False,          # IterBased 修改 !
        milestones=[5],
        gamma=0.1)
]

train_dataloader = dict(
    sampler=dict(type='InfiniteSampler'),
    dataset=dict(
        ann_file='reid/meta/train.txt',
        triplet_sampler=dict(num_ids=16, ins_per_id=4),
))
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=10000,
        save_best='auto'
    )
    # checkpoint=dict(type='CheckpointHook', interval=1,save_best='auto'),
)


train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=70938,
    # val_interval=500,
)
# train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')