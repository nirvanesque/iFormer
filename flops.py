import torch
import time
from timm import create_model
import models
import utils
from fvcore.nn import FlopCountAnalysis

T0 = 5
T1 = 10

for n, batch_size, resolution in [
    ('iFormer_l0', 1024, 224),
]:
    inputs = torch.randn(1, 3, resolution,
                            resolution)
    model = create_model(n, num_classes=1000, drop_path_rate=0., layer_scale_init_value=1e-6)
    for key, item in model.named_parameters():print(key)
    model.eval()
    print(model)
    utils.replace_batchnorm(model)
    # print(model)
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters / 1e6)
    flops = FlopCountAnalysis(model, inputs)
    print("flops: ", flops.total() / 1e9)