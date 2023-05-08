import socket
import pickle
import torch
class ModelClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self):
        self.sock.connect((self.host, self.port))

    def get_state_dict(self):
        # Receive the pickled state_dict from the server
        packed_state_dict = b''
        while True:
            chunk = self.sock.recv(1024)
            if not chunk:
                break
            packed_state_dict += chunk

        # Unpickle the state_dict and return it
        state_dict = pickle.loads(packed_state_dict)
        return state_dict

    def close(self):
        self.sock.close()

import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 2),
    nn.Sigmoid()
)

client = ModelClient('localhost', 8000)
client.connect()
state_dict = client.get_state_dict()
print(state_dict)
client.close()
