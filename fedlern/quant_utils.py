import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# def create_model(num_classes=10):

#     model = ResNet(in_channels=16, num_classes=num_classes)
#     return model

class QuantizedResNet(nn.Module):
    def __init__(self, model_fp32):
        super(QuantizedResNet, self).__init__()
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        # FP32 model
        self.model_fp32 = model_fp32

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.model_fp32(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x

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

def get_model2(model, learning_rate=1e-3, weight_decay=1e-4):

    # set the first layer not trainable
    # model.features.conv0.weight.requires_grad = False

    # all fc layers
    weights = [
        p for n, p in model.named_parameters()
        if 'weight' in n and 'conv' not in n
    ]

    # all conv layers
    weights_to_be_quantized = [
        p for n, p in model.named_parameters()
        # if 'conv' in n and 'conv0' not in n
        if 'conv' in n and 'weight' in n
    ]

    biases = [
        p for n, p in model.named_parameters()
        if 'bias' in n
    ]    

    params = [
        {'params': weights, 'weight_decay': weight_decay},
        {'params': weights_to_be_quantized, 'weight_decay': weight_decay},
        {'params': biases,  'weight_decay': weight_decay}
    ]
    optimizer = optim.SGD(params, lr=learning_rate, momentum=0.9)

    loss = nn.CrossEntropyLoss().cuda()
    model = model.cuda()  # move the model to gpu
    return model, loss, optimizer

def quantize_bw(kernel : torch.Tensor):
    """
    binary quantization
    Return quantized weights of a layer.
    """
    delta = kernel.abs().mean()
    sign = kernel.sign().float()

    return sign*delta