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
            size_bytes = s.recv(4)
            size = int.from_bytes(size_bytes, byteorder='big')
            model_bytes = b''
            while len(model_bytes) < size:
                model_bytes += s.recv(4096)
            self.local_model += torch.deserialize(model_bytes)
            print('Global model received & deserialized')

            # Train local model
            self.train()
            
            # Send local model to server
            local_model_bytes = torch.serialize(self.local_model)
            # Send size of local model first to handle truncated data
            size_bytes = len(local_model_bytes).to_bytes(4, byteorder='big')
            s.sendall(size_bytes)
            s.sendall(local_model_bytes)
            print('Local model sent to server')

    def train(self):
        # Train local model with local data
        pass

if __name__ == '__main__':
    client = Client()
    client.start()
