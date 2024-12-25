
_base_ = ['./reid_pmnet_4xb32_14000iter_mot17train80_test-mot17val20.py']

model = dict(head=dict(num_classes=1701))
# data
data_root = 'data/MOT20/'
train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader = dict(dataset=dict(data_root=data_root))
test_dataloader = val_dataloader