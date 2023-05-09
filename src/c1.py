import socket
import torch

HOST = '127.0.0.1'  # Server IP address
PORT = 65432  # Port to connect to

class Client:
    def __init__(self):
        self.local_model = torch.rand(3, 3)  # initialize local model

    def start(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            print('Connected to server')

            # Receive global model from server
            global_model = s.recv(4096)
            self.local_model += torch.deserialize(global_model)
            print('Global model received')

            # Train local model
            self.train()
            
            # Send local model to server
            s.sendall(torch.serialize(self.local_model))
            print('Local model sent to server')

    def train(self):
        # Train local model with local data
        pass

if __name__ == '__main__':
    client = Client()
    client.start()
