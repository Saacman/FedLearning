#!/usr/bin/env python
# coding: utf-8

# # Federated Learning + Linear Quantization for RESNET18 & CIFAR10

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import numpy as np
#from tqdm import tqdm
import seaborn as sns
from fedlern.utils import *
from fedlern.train_utils import *
from fedlern.quant_utils import *
from fedlern.models.resnet_v2 import ResNet18
import csv
import argparse


def main(id: int, bits: int):


    epoch = 5
    rounds = 30
    num_clients = 10
    lrn_rate = 0.1
    clients_sample_size = int(.3 * num_clients) # Use 30% of available clients
    num_workers = 8
    train_batch_size = 32 
    eval_batch_size= 3
    stats = (0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)
    #quantize_bits = 4
    id = id
    quantize_bits = bits

    csv_file_path = "./report.csv"



    # Data augmentation and normalization for training
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
    ])

    # Normalization for testing
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*stats)
    ])

    # CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform_test)

    # split the training data
    train_splits = torch.utils.data.random_split(train_dataset, [int(train_dataset.data.shape[0]/num_clients) for i in range(num_clients)])

    # Data loaders
    train_loaders = [DataLoader(dataset=split, batch_size=train_batch_size, shuffle=True, num_workers=num_workers) for split in train_splits]
    test_loader = DataLoader(dataset=test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    global_model = ResNet18().to(device)
    client_models = [ResNet18()
                        .to(device)
                        #.load_state_dict(global_model.state_dict())
                    for _ in range(num_clients)]


    # # Define the criterion, and pair of optimizers for global model
    # The first optimizer holds the regular weights, while the second one holds the quantized parameters, which are to be propagated to the clients.

    # In[5]:


    # Optimizer & criterion based on gobal model
    global_optimizer = get_model_optimizer(global_model,
                                        learning_rate=lrn_rate,
                                        weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss().to(device) # computes the cross-entropy loss between the predicted and true labels

    # ----Optimizer for quantized model on global model----
    # Copy the parameters
    all_G_kernels = [kernel.data.clone().requires_grad_(True)
                    for kernel in global_optimizer.param_groups[1]['params']]

    kernels = [{'params': all_G_kernels}]

    # New optimizer for the quantized weights
    goptimizer_quant = optim.SGD(kernels, lr=0)





    # ## Define optimizers for the clients.
    # Each clients has a pair of optimizers

    optimizers = [get_model_optimizer(model,learning_rate=lrn_rate, weight_decay=5e-4) for model in client_models]


    optimizers_quant = []
    for optimizer in optimizers:
        # Copy the parameters
        all_G_kernels = [kernel.data.clone().requires_grad_(True)
                        for kernel in optimizer.param_groups[1]['params']]

        # Handle of the optimizer parameters
        all_W_kernels = optimizer.param_groups[1]['params']
        kernels = [{'params': all_G_kernels}]

        # New optimizer for the quantized weights
        optimizer_quant = optim.SGD(kernels, lr=0)
        optimizers_quant.append(optimizer_quant)



    server_test = to_quantized_model_decorator(global_optimizer, goptimizer_quant)(evaluate_model)

    # Decorators magic
    # first we wrap the update function with the decorator for the global model switch
    server_switch_aggregate = to_quantized_model_decorator(global_optimizer, goptimizer_quant)(server_update)
    # then we wrap the update function with the decorator for each of the clients switch
    switch_client_weights = [to_quantized_model_decorator(fp32_optimizer, quant_optimizer) for fp32_optimizer, quant_optimizer in zip(optimizers, optimizers_quant)]



    server_update_quant  = switch_client_weights[0](server_switch_aggregate)
    for funct in switch_client_weights[1:]:
        server_update_quant = funct(server_update_quant)




    server_update_quant(global_model, client_models)



    # initialize lists to store the training and testing losses and accuracies
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    #for round in tqdm(range(rounds)):
    for round in range(rounds):


        # Select n random clients
        selected_clients = np.random.permutation(num_clients)[:clients_sample_size]
        # Train the selected clients
        for client in selected_clients:
            # Individual criterion and optimizer
            print(f"Client {client} training")
            train_loss, train_acc = qtrain_model(model = client_models[client],
                                                train_loader = train_loaders[client],
                                                device = device,
                                                criterion = criterion,
                                                optimizer = optimizers[client],
                                                optimizer_quant = optimizers_quant[client],
                                                num_epochs=epoch,
                                                bits = quantize_bits)

        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Aggregate in 3 steps
        server_aggregate(global_model, client_models)
        server_quantize(global_optimizer, goptimizer_quant, quantize_bits)
        server_update_quant(global_model, client_models)


        # Test the global model
        test_loss, test_acc = server_test(model=global_model,
                                        test_loader=test_loader,
                                        device=device,
                                        criterion=criterion)

        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f"{round}-th ROUND: average train loss {(train_loss / clients_sample_size):0.3g} | test loss {test_loss:0.3g} | test acc: {test_acc:0.3f}")
        
        

    quantized_histogram = to_quantized_model_decorator(global_optimizer, goptimizer_quant)(histogram_conv1)

    # # Evaluate the model's histogram
    # before = histogram_conv1(global_model)
    # plot_histogram(before[0], before[1],f"CONV1 - fp32", f"./plots/")


    after = quantized_histogram(global_model)
    plot_histogram(after[0], after[1], f"CONV1 - {quantize_bits} bits", save=True, save_path=f"./plots/histogram_conv1_{id}_{quantize_bits}.png")



    # plot the training loss
    sns.set(style='darkgrid')
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.show()
    plt.savefig(f"./plots/training_loss_{id}_{quantize_bits}_bits.png")


    train_accs = [d.item() for d in train_accs]
    test_accs = [d.item() for d in test_accs]



    # plot the training and testing accuracies
    sns.set(style='darkgrid')
    plt.plot(train_accs, label='Train Acc')
    plt.plot(test_accs, label='Test Acc')
    plt.title('Training and Testing Accuracies')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    #plt.show()
    plt.savefig(f"./plots/accuracy_{id}_{quantize_bits}_bits.png")


    # Test quantized model
    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")
    # Test the model
    loss, acc = server_test(global_model, test_loader, device,criterion=criterion)
    print(f'Loss: {loss}, Accuracy: {acc*100}%')
    print_model_size(global_model)

    # Measure inference latency
    cpu_inference_latency = measure_inference_latency(model=global_model, device=cpu_device, input_size=(1,3,32,32), num_samples=100)
    gpu_inference_latency = measure_inference_latency(model=global_model, device=cuda_device, input_size=(1,3,32,32), num_samples=100)
    print("CPU Inference Latency: {:.2f} ms / sample".format(cpu_inference_latency * 1000))
    print("CUDA Inference Latency: {:.2f} ms / sample".format(gpu_inference_latency * 1000))



    with open(csv_file_path, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # Write the data to the CSV file
        new_data = [id, quantize_bits, loss, acc]
        csv_writer.writerow(new_data)

if __name__ == "__main__":
    main()