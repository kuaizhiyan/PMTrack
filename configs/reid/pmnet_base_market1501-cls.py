_base_ = [
    '../_base_/datasets/market1501-cls.py',
    '../_base_/schedules/imagenet_bs2048_AdamW.py',
    '../_base_/default_runtime.py'
]


model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='mmpretrain.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1,2,3),
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
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
    ),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=256,
        init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False),
    
    )

train_dataloader = dict(
    batch_size=256,
    # num_workers=5,
    # dataset=dict(
    #     type=dataset_type,
    #     data_root='data/imagenet',
    #     split='train',
    #     pipeline=train_pipeline),
    # sampler=dict(type='DefaultSampler', shuffle=True),
)
val_dataloader = dict(
    batch_size=256,
    # num_workers=5,
    # dataset=dict(
    #     type=dataset_type,
    #     data_root='data/imagenet',
    #     split='val',
    #     pipeline=test_pipeline),
    # sampler=dict(type='DefaultSampler', shuffle=False),
)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=320, val_interval=2)
val_cfg = dict()
test_cfg = dict()

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.0015, weight_decay=0.3),
    # specific to vit pretrain
    paramwise_cfg=dict(custom_keys={
        '.cls_token': dict(decay_mult=0.0),
        '.pos_embed': dict(decay_mult=0.0)
    }),
)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=256)
default_hooks = dict(
    # save checkpoint per epoch.
    # checkpoint=dict(type='CheckpointHook', interval=2),
    checkpoint=dict(type='CheckpointHook', interval=2,save_best='auto'),
)
# visualizer = dict(
    
# )
