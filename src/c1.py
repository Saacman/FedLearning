import socket
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pickle
import models 


HOST = '127.0.0.1'  # Server IP address
PORT = 65432  # Port to connect to

epoch = 5
num_clients = 10
sample_size = int(.3 * num_clients) # Use 30% of available clients
net_parameters = [ 28 * 28, # input
                512, 256, 128, 64,
                10 ] #output

class Client:
    def __init__(self, data_loader, device):
        self.local_model = models.MLP(net_parameters)  # initialize local model
        self.epoch = 5
        # TODO: Set optimizer and criterion with decorators
        self.optimizer = optim.Adam(self.local_model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.data_loader = data_loader
        self.device = device

    def start(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            print('Connected to server')

            # Receive global model size from server
            size_bytes = b''
            while len(size_bytes) < 4:
                size_bytes += s.recv(4 - len(size_bytes))
            size = int.from_bytes(size_bytes, byteorder='big')
            # Receive global model from server
            model_bytes = b''
            while len(model_bytes) < size:
                model_bytes += s.recv(4096)

            recvd_model_dict = pickle.loads(model_bytes)
            print('Global model received & deserialized')

            # Train local model
            self.local_model.to(self.device) # TODO: maybe move to start?
            self.local_model.load_state_dict(recvd_model_dict)
            self.train()
            
            # Send local model to server
            local_model_bytes = pickle.dumps(self.local_model.state_dict())
            # Send size of local model first to handle truncated data
            size_bytes = len(local_model_bytes).to_bytes(4, byteorder='big')
            s.sendall(size_bytes)
            s.sendall(local_model_bytes)
            print('Local model sent to server')

    def train(self):
        
        self.local_model.train()
        for e in range(self.epoch):
            for images, labels in self.data_loader:
                images, labels = images.to(device), labels.to(device)
                self.optimizer.zero_grad() # reset the gradients to zero
                output = self.local_model(images) # forward pass
                loss = self.criterion(output, labels) # compute the loss
                loss.backward() # compute the gradients
                self.optimizer.step() # update the parameters

        return loss.item() * images.size(0)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define transformation to apply to each image in the dataset
    transform = transforms.Compose([
        transforms.ToTensor(), # convert the image to a PyTorch tensor
        transforms.Normalize((0.5,), (0.5,)) # normalize the image with mean=0.5 and std=0.5
    ])

    # load the MNIST training and testing datasets
    train_dataset = datasets.MNIST(root='data/', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='data/', train=False, transform=transform, download=True)

    # split the training data
    train_split = torch.utils.data.random_split(train_dataset, [int(train_dataset.data.shape[0]/num_clients) for i in range(num_clients)])

    # create data loaders to load the datasets in batches during training and testing
    train_loader = [torch.utils.data.DataLoader(split, batch_size=64, shuffle=True) for split in train_split]
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    for data_ld in train_loader:
        client = Client(data_ld, device)
        client.start()
