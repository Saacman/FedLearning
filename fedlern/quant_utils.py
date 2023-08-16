import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from fedlern.quantize import *
import matplotlib.pyplot as plt
from fedlern.train_utils import *
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

def get_model_optimizer(model, learning_rate=1e-3, weight_decay=1e-4):

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

    #loss = nn.CrossEntropyLoss().cuda()
    #model = model.cuda()  # move the model to gpu
    return optimizer

def quantize_bw(kernel : torch.Tensor):
    """
    binary quantization
    Return quantized weights of a layer.
    """
    delta = kernel.abs().mean()
    sign = kernel.sign().float()

    return sign*delta


def qtrain_model(model : torch.nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 device, criterion = None, optimizer = None, optimizer_quant = None, scheduler = None,
                 num_epochs=20, learning_rate=1e-2, momentum=0.9, weight_decay=1e-5,
                 bits = 8, eta = 1):
    
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    if scheduler is None:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120, 160], gamma=0.1)

    # Copy the parameters
    all_G_kernels = [kernel.data.clone().requires_grad_(True)
                    for kernel in optimizer.param_groups[1]['params']]
    
    # Handle of the optimizer parameters
    all_W_kernels = optimizer.param_groups[1]['params']
    # kernels = [{'params': all_G_kernels}]

    # # New optimizer for the quantized weights
    # optimizer_quant = optim.SGD(kernels, lr=0)

    # Training
    model.to(device)
    model.train()
    for epoch in range(num_epochs):


        running_loss = 0
        running_corrects = 0

        for inputs, labels in train_loader:
            #before = histogram_conv1(model)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # Get a handle of the parameters
            all_W_kernels = optimizer.param_groups[1]['params']
            all_G_kernels = optimizer_quant.param_groups[0]['params']

            for k_W, k_G in zip(all_W_kernels, all_G_kernels):
                V = k_W.data                
                # -- Apply quantization
                k_G.data = quantize(V, num_bits=bits)

                # -- Switch the weights
                k_W.data, k_G.data = k_G.data, k_W.data
            #after = histogram_conv1(model)
            # forward + backward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # -- Switch the weights back
            for k_W, k_G in zip(all_W_kernels, all_G_kernels):
                k_W.data, k_G.data = k_G.data, k_W.data


            _, preds = torch.max(outputs, 1)
            # -- Step the optimizer
            optimizer.step()
            #scheduler.step()

            # -- statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = running_corrects / len(train_loader.dataset)

        print(f"Epoch: {epoch}/{num_epochs} Train Loss: {train_loss:.3f} Train Acc: {train_accuracy:.3f}")

    #return model
    return train_loss, train_accuracy#, before, after


def to_quantized_model_decorator(optimizer, optimizer_quant):
    def decorator(func):
        def wrapper(*args, **kwargs):
            #print("Wrapper used")
            # -- Get a handle of the parameters
            all_W_kernels = optimizer.param_groups[1]['params']
            all_G_kernels = optimizer_quant.param_groups[0]['params']
            # -- Switch the weights
            for k_W, k_G in zip(all_W_kernels, all_G_kernels):
                k_W.data, k_G.data = k_G.data, k_W.data

            # Call the original function
            result = func(*args, **kwargs)

            # Code block after the function's code execution
            # -- Switch the weights back
            for k_W, k_G in zip(all_W_kernels, all_G_kernels):
                k_W.data, k_G.data = k_G.data, k_W.data
            #print("Exit wrapper")
            return result
        return wrapper
    return decorator

def server_aggregate(global_model : torch.nn.Module, client_models: list):
    """
    The means of the weights of the client models are aggregated to the global model
    """
    global_dict = global_model.state_dict() # Get a copy of the global model state_dict
    for key in global_dict.keys():
        global_dict[key] = torch.stack([client_models[i].state_dict()[key].float() for i in range(len(client_models))],0).mean(0)
    global_model.load_state_dict(global_dict)
    
    # Update the client models using the global model
    for model in client_models:
        model.load_state_dict(global_model.state_dict())

def server_update(global_model : torch.nn.Module, client_models: list):
    for model in client_models:
        model.load_state_dict(global_model.state_dict())

def server_quantize(optimizer, optimizer_quant, bits=8):
    # Get a handle of the parameters
    all_W_kernels = optimizer.param_groups[1]['params']
    all_G_kernels = optimizer_quant.param_groups[0]['params']
    
    for k_W, k_G in zip(all_W_kernels, all_G_kernels):
        V = k_W.data                
        # -- Apply quantization
        #print("Before quantization---------------")
        #print(k_W.data)
        k_G.data = quantize(V, num_bits=bits)
        #print("After quantization---------------")
        #print(k_G.data)
        # -- Switch the weights
        k_W.data, k_G.data = k_G.data, k_W.data
    
    # -- Switch the weights back
    for k_W, k_G in zip(all_W_kernels, all_G_kernels):
        k_W.data, k_G.data = k_G.data, k_W.data


def plot_histogram(hist, bin_edges, title='Histogram', save: bool = False, save_path: str = None):
    plt.bar(bin_edges[:-1], hist, width=(bin_edges[1] - bin_edges[0]), align='edge')
    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    plt.title(title)
    #plt.show()
    if save:
        plt.savefig(save_path)
    else:
        plt.show()

def histogram_conv1(model : torch.nn.Module):
    '''
    Inspect a tensor from the model's state dictionary
    '''
    state_dict = model.state_dict()

    # Choose the key corresponding to the element you want to print
    # For example, let's say you want to print the weights of a specific layer 'conv1'
    element_key = 'conv1.weight'

    # Check if the key exists in the state dictionary
    if element_key in state_dict:
        element = state_dict[element_key]
        return torch.histogram(element.to(torch.device('cpu')))
    else:
        print(f"The key '{element_key}' does not exist in the model's state dictionary.")
