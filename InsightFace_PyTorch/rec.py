import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import argparse
from config import device, grad_clip, print_freq, num_workers, logger
from data_gen import ArcFaceDataset
from focal_loss import FocalLoss
from megaface_eval import megaface_test
from models import resnet18, resnet34, resnet50, resnet101, resnet152, ArcMarginModel
from utils import parse_args, save_checkpoint, AverageMeter, accuracy, clip_gradient