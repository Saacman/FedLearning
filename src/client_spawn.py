import socket, pickle
import torch
import torch.nn as nn
import torch.optim as optim
from model import MLP

epoch = 5
comm_cycles = 25
num_clients = 10
sample_size = int(.3 * num_clients) # Use 30% of available clients
net_parameters = [ 28 * 28, # input
                512, 256, 128, 64,
                10 ] #output

def client_update(  client_model : torch.nn.Module,
                    optimizer,
                    criterion,
                    data_loader : torch.utils.data.DataLoader,
                    device,
                    epoch = 5):
    """
    Train the client model on client data
    """
    client_model.train()
    for e in range(epoch):
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad() # reset the gradients to zero
            output = client_model(images) # forward pass
            loss = criterion(output, labels) # compute the loss
            loss.backward() # compute the gradients
            optimizer.step() # update the parameters

    return loss.item() * images.size(0)

if __name__ == '__main__':
    # Socket instantiation
    
    HOST = 'localhost'
    PORT = 50009
    socketClient = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socketClient.connect((HOST, PORT))

    # Model instantiation
    client_models = [MLP(net_parameters) for _ in range(num_clients)]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Here we must receive the model from the server
    #global_model.to(device)

    for model in client_models:
        model.to(device)
        #model.load_state_dict(global_model.state_dict())
        arr = "init"
        data_string = pickle.dumps(arr)
        socketClient.send(data_string)
        data = []
        while(True):
            packet = socketClient.recv(4096)
            if not packet: break
            data.append(packet)
        
        gstate = pickle.loads(b"".join(data))
        socketClient.close()
        print ('Received', repr(gstate))
        model.load_state_dict(gstate)

    criterion = nn.CrossEntropyLoss() # computes the cross-entropy loss between the predicted and true labels
    optimizers =[optim.Adam(model.parameters(), lr=0.001) for model in client_models]
    