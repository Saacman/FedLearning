import torch
import torch.nn as nn
import os
import copy
import numpy as np

from torch.quantization import quantize_dynamic

from src.models import ResNet
from src import utils as u

random_seed = 0
num_classes = 10
cuda_device = torch.device("cuda:0")
cpu_device = torch.device("cpu:0")
def model_equivalence(model_1, model_2, device, rtol=1e-05, atol=1e-08, num_tests=100, input_size=(1,3,32,32)):

    model_1.to(device)
    model_2.to(device)

    for _ in range(num_tests):
        x = torch.rand(size=input_size).to(device)
        y1 = model_1(x).detach().cpu().numpy()
        y2 = model_2(x).detach().cpu().numpy()
        if np.allclose(a=y1, b=y2, rtol=rtol, atol=atol, equal_nan=False) == False:
            print("Model equivalence test sample failed: ")
            print(y1)
            print(y2)
            return False

    return True

model_dict = torch.load('models/resnet.ckpt')
model = ResNet(in_channels=16, num_classes=10)
model.load_state_dict(model_dict)
model.to(cpu_device)
# Make a copy of the model for layer fusion
fused_model = copy.deepcopy(model)

model.eval()
# The model has to be switched to evaluation mode before any layer fusion.
# Otherwise the quantization will not work correctly.
fused_model.eval()

# Fuse the model in place rather manually.
fused_model = torch.quantization.fuse_modules(fused_model, [["conv1", "bn1", "relu"]], inplace=True)
for module_name, module in fused_model.named_children():
    print(module_name)
    if "layer" in module_name:
        for basic_block_name, basic_block in module.named_children():
            print(f"\t{basic_block_name}")
            torch.quantization.fuse_modules(basic_block, [["conv1", "bn1", "relu"], ["conv2", "bn2"]], inplace=True)
            for sub_block_name, sub_block in basic_block.named_children():
                print(f"\t\t{sub_block_name}")
                if sub_block_name == "shortcut":
                    torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)

# Print FP32 model.
print("FP32", model)
# Print fused model.

print("FUSED", fused_model)

assert model_equivalence(model_1=model, model_2=fused_model, device=cpu_device, rtol=1e-03, atol=1e-06, num_tests=100, input_size=(1,3,32,32)), "Fused model is not equivalent to the original model!"