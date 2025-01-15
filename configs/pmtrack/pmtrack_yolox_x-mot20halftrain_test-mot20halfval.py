_base_ = [
    './pmtrack_yolox_x-mot17halftrain_test-mot17halfval.py',  # noqa: E501
]
data_root='./data/MOT20/'

train_dataloader = None
val_dataloader = dict(
    # Now StrongSORT only support video_based sampling
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=_base_.data_root,
        ann_file='annotations/half-val_cocoformat.json',
        data_prefix=dict(img_path='train'),
        # when you evaluate track performance, you need to remove metainfo
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader




