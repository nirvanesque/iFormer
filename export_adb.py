import os

import torch

from timm import create_model
import models

import utils

import torch
import torchvision
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('--model', default='iFormer_t', type=str)
parser.add_argument('--resolution', default=224, type=int)
parser.add_argument('--ckpt', default=None, type=str)

if __name__ == "__main__":
    # Load a pre-trained version of MobileNetV2
    args = parser.parse_args()
    model = create_model(args.model, num_classes=1000)
    utils.replace_batchnorm(model)
    model.eval()

    # Trace the model with random data.
    resolution = args.resolution
    example_input = torch.rand(1, 3, resolution, resolution) 
    traced_model = torch.jit.trace(model, example_input)

    # Save the converted model.
    # save_name = f"output/adb/{args.model}_{resolution}.ptl"
    save_name = f"./{args.model}_{resolution}.ptl"
    # traced_model.save(save_name)
    traced_model._save_for_lite_interpreter(save_name)
    print(save_name)