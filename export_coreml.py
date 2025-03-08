import torch

from timm import create_model
import models

import utils

import torch
import torchvision
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('--model', default='iFormer_l0', type=str)
parser.add_argument('--resolution', default=224, type=int)
parser.add_argument('--ckpt', default=None, type=str)

if __name__ == "__main__":
    # Load a pre-trained version of MobileNetV2
    args = parser.parse_args()
    if args.model == 'iFormer_h':
        model = create_model(args.model, num_classes=1000, drop_path_rate=0., layer_scale_init_value=1e-6)
    else:
        model = create_model(args.model, num_classes=1000, drop_path_rate=0., layer_scale_init_value=0.)
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt)['model'])
    utils.replace_batchnorm(model)
    model.eval()

    # Trace the model with random data.
    resolution = args.resolution
    example_input = torch.rand(1, 3, resolution, resolution) 
    traced_model = torch.jit.trace(model, example_input)
    out = traced_model(example_input)

    import coremltools as ct

    # Using image_input in the inputs parameter:
    # Convert to Core ML neural network using the Unified Conversion API.
    model = ct.convert(
        traced_model,
        inputs=[ct.ImageType(shape=example_input.shape)]
    )

    # Save the converted model.
    save_name = f"./{args.model}_{resolution}.mlpackage"
    model.save(save_name)
    print(save_name)
