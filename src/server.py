import socket, pickle
import torch
import torch.nn as nn
import torch.optim as optim
from model import MLP

net_parameters = [ 28 * 28, # input
                512, 256, 128, 64,
                10 ] #output

def server_aggregate(global_model : torch.nn.Module, client_models):
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


if __name__ == '__main__':
    # Socket instantiation
    HOST = 'localhost'
    PORT = 50009
    socketServer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socketServer.bind((HOST, PORT))
    socketServer.listen(5)

    # Global Model instantiation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global_model = MLP(net_parameters)
    global_model.to(device)
    conn, addr = socketServer.accept()
    print ('Connected by', addr)
    while True:
        data = conn.recv(4096)
        print(pickle.loads(data))
        print(not data)
        if not data:
            print(not data)
            break

    current_state = pickle.dumps(global_model.state_dict())
    print(len(current_state))
    print("Sending model")
    conn.send(current_state)
    print("Data sent")
    while True:
        data = conn.recv(4096)
        print(pickle.loads(data))
        if not data: break
    conn.close()
