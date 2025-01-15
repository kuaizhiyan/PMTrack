# Installation

PMTrack builds upon [mmdetection](https://github.com/open-mmlab/mmdetection/tree/main). We use Python 3.8.18, pytorch 1.12.1 and cuda 11.3 for experiments.


## Clone the Repository
First, clone the repository to your local machine:
```bash
git clone https://github.com/kuaizhiyan/PMTrack.git
cd PMTrack
```

## Set up the Enviroment
It is recommended to use a virtual environment (e.g., conda or venv) to manage dependencies.
```bash
conda create -n pmtrack python=3.8 -y
conda activate pmtrack
pip install -r requirements.txt
```
Using requirements.txt:
The requirements.txt should contain the necessary dependencies for the project, for example:

```plaintext
pytorch>=1.12.1
mmcv>=2.0.1
mmengine>=0.9.0
mmpretrain>=1.2.0
```
Run the following command in the project root directory:
```bash
pip install -e .
```
