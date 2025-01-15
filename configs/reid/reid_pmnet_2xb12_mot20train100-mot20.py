_base_ = ['./reid_pmnet_2xb12_mot17train100-mot17.py']

pretrained='/home/kzy/project/PartDecoder/mmdetection/work_dirs/reid_pmnet_2xb12_mot17train100-mot17/best_reid-metric_mAP_iter_39000.pth'
model = dict(head=dict(num_classes=2130),
             init_cfg=dict(
                 type='Pretrained',
                 checkpoint=pretrained,
             ))
# data
data_root = 'data/MOT20/'
train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader = dict(dataset=dict(data_root=data_root))
test_dataloader = val_dataloader


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