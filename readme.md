# Bredvid reinforcement learning for tetris

## Setup
- Clone the repo
- `cd Bredvid-RL-Tetris`

Download python 3.7.7 or greater e.g. [here](https://www.python.org/downloads/)  
- Make sure you have pip installed if python is downloaded from other sources  
  
Download pytorch [here](https://pytorch.org/get-started/locally/), with: 
- stable build
- Package: pip
- language: Python
- CUDA: 10.2  


Packages  
a. `pip install virtualenv`  
b. `python -m virtualenv venv`  
c1. Windows: `.\venv\Scripts\activate`    
c2. Unix: `source venv/bin/activate`  
d1. Mac/Linux:  `pip install torch torchvision`
d2. Windows: `pip install torch===1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html`
d. `pip install -r requirements.txt`  
e. `python setupCheck.py`  
- Follow eventual prompts
- If you have a graphics card, but don't have CUDA, install it here:
  - [Windows og Linux](
https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exenetwork)
  - [OS-X](https://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html)
- If SummaryWriter doesn't work, double check that only `tensorboardX` is installed and not `tensorboard` or other implementations by running `pip list`

## Course Contents
Deep Q-learning for playing Tetris
-

<p align="center">
  <img src="demo/tetris.gif" width=600><br/>
  <i>Tetris demo</i>
</p>


## Attributions
Viet Nguyen nhviet1009@gmail.com  
https://github.com/uvipen/Tetris-deep-Q-learning-pytorch/blob/master/train.py 
