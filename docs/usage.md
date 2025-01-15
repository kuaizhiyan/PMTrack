# Usage

We provide commands for single GPU (single machine) and multi-GPU (single machine) training and testing. For detailed methods and explanations, please refer to the [MMDetection official documentation](https://mmpretrain.readthedocs.io/en/stable/user_guides/train.html).
## Train

### Train with your PC
You can use tools/train.py to train a model on a single machine with a CPU and optionally a GPU.

Here is the full usage of the script:
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py ${CONFIG_FILE} [ARGS]
```
For example, you can train PMTrack by:
```bash
 python tools/train.py configs/reid/reid_pmnet_2xb32_mot17train80_test-mot17val20.py
```

### Train with mutiple GPUs
If you want to startup multiple training jobs and use different GPUs, you can launch them by specifying different ports and visible devices.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash ./tools/dist_train.sh ${CONFIG_FILE1} 4 [PY_ARGS]
```
You can train PMTrack by:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash ./tools/dist_train.sh configs/reid/reid_pmnet_2xb32_mot17train80_test-mot17val20.py 2
```

## Test

### Test with your PC
You can use tools/test.py to test a model on a single machine with a CPU and optionally a GPU.

Here is the full usage of the script:
```bash
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [ARGS]
```

You can test by:
```bash
python tools/test.py configs/reid/reid_pmnet_2xb32_mot17train80_test-mot17val20.py
```

### Test with multiple GPUs
We provide a shell script to start a multi-GPUs task with torch.distributed.launch.
```bash
bash ./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [PY_ARGS]
```

For example you can test with multiple GPUs by:
```bash
CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_test.sh configs/reid/reid_pmnet_2xb32_mot17train80_test-mot17val20.py 2
```

### Test tracking
- You can set the results saving path by modifying the key outfile_prefix in evaluator. For example, val_evaluator = dict(outfile_prefix='results/sort_mot17'). Otherwise, a temporal file will be created and will be removed after evaluation.
- If you just want the formatted results without evaluation, you can set format_only=True. For example, test_evaluator = dict(type='MOTChallengeMetric', metric=['HOTA', 'CLEAR', 'Identity'], outfile_prefix='sort_mot17_results', format_only=True)

If you want to test the model on single GPU, you can directly use the tools/test_tracking.py as follows.

```bash
python tools/test_tracking.py configs/pmtrack/pmtrack_yolox_x-mot17_test.py
```

python tools/test_tracking.py `${CONFIG_FILE}` [optional arguments]
You can use export `CUDA_VISIBLE_DEVICES=$GPU_ID` to select the GPU.

```bash
bash ./tools/dist_test_tracking.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```
An example of testing the MOT model DeepSort on single node multiple GPUs:
```bash
CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_test_tracking.sh PMTrack/configs/pmtrack/pmtrack_yolox_x-mot17_test.py 2
```

