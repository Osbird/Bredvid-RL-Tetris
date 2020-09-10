import torch
import torch.nn as nn
import pygame
import tqdm
import cv2
from tensorboardX import SummaryWriter
import matplotlib
import os
import argparse
import shutil
from random import random, randint, sample
from PIL import Image
import numpy as np

if torch.cuda.is_available():
    print("You have CUDA, great!")
else: 
    print("CUDA is not enabled on your device. \nIf you don't have a graphics card that's just as expected! \nIf you do have a graphics card on your machine, install it from the links in the readme")

log_path = "tensorboard"
os.makedirs(log_path, exist_ok=True)
writer = SummaryWriter(log_path)

try:
    writer.add_scalar('SetupCheck', 0, 0)
except Exception as ex:
    print("SummaryWriter does not work \nThe error message was: ", ex)
    print("Potentially follow the commands in the readme") 
else:
    print("SummaryWriter works, great!")

