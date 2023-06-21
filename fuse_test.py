import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fedlern.models.resnet_v2 import BasicBlock, Bottleneck, ResNet18

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
        


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#model_dict = torch.load('saved_models/resnet18_cifar10_92-2.pt')
# Create an instance of the ResNet18 model
model = ResNet18()
#model.load_state_dict(model_dict)
fused_model = ResNet18()
#fused_model.load_state_dict(model_dict)


print(model)
model.eval()
fused_model.eval()
# Fuse the first Conv2d, BatchNorm2d, and ReLU layers
fused_model = torch.quantization.fuse_modules(model, [["conv1", "bn1", "relu"]], inplace=False)
# Fuse the remaining Conv2d and BatchNorm2d layers in the ResNet blocks
for name, module in fused_model.named_modules():
    if isinstance(module, nn.Sequential):
        for i in range(len(module)):
            if isinstance(module[i], BasicBlock) or isinstance(module[i], Bottleneck):
                module[i] = torch.quantization.fuse_modules(module[i], [["conv1", "bn1", "relu"], ["conv2", "bn2"]], inplace = False)


# Print FP32 model.
print("FP32", model)

# Print fused model.
print("FUSED", fused_model)

assert model_equivalence(model_1=model, model_2=fused_model, device=device, rtol=1e-03, atol=1e-06, num_tests=100, input_size=(1,3,32,32)), "Fused model is not equivalent to the original model!"

