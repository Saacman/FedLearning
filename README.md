# Federated Binary Quantization for ResNet18

This repository presents an innovative approach to enhancing machine learning models by combining federated learning and binary quantization techniques. The project revolves around a ResNet18 network and investigates the potential benefits of integrating these two methodologies.

## Introduction

### Federated Learning
Federated learning is a distributed approach to training machine learning models. Unlike traditional centralized training, where data is collected and processed in a single location, federated learning allows training on data distributed across multiple devices or servers while keeping the data decentralized. This approach minimizes data sharing, making it particularly useful for scenarios where data privacy and security are paramount.

### Binary Quantization
Binary quantization is a technique used to reduce the memory and computation requirements of neural networks. Instead of representing weights and activations using traditional floating-point numbers, binary quantization maps them to binary values (usually -1 or +1). This reduces memory usage, accelerates computations, and often leads to faster inference.

## Benefits of Mixing Federated Learning and Binary Quantization

Combining federated learning with binary quantization introduces several advantages:

1. **Enhanced Privacy**: Federated learning inherently preserves data privacy since data remains on local devices. By integrating binary quantization, the model's weight updates exchanged during federated learning can be even more privacy-preserving, as they involve transmitting binary values instead of real numbers.

2. **Reduced Communication Overhead**: In federated learning, communication between the central server and devices can be a bottleneck. By using binary quantization, the size of data transmitted during communication is significantly reduced, resulting in lower communication overhead.

3. **Efficient Resource Utilization**: Binary quantization considerably reduces memory usage and computation requirements. This efficiency becomes crucial in federated settings where devices might have limited computational resources.

4. **Faster Convergence**: Binary quantization can lead to faster convergence during training. When integrated into the federated learning paradigm, this can potentially speed up the convergence of global models across devices.


### Accuracy Analysis

![Accuracy Chart](link_to_accuracy_chart.png)
*Figure 1: Model accuracy with respect to different bit lengths.*

### Architecture Diagram

![Architecture Diagram](link_to_architecture_diagram.png)
*Figure 2: Illustration of the Federated Binary Quantization setting.*

### Value Distribution

![Histogram](link_to_histogram.png)
*Figure 3: Histogram showing value distribution in convolutional layers for different bit lengths.*

## Usage