import socket
import pickle
import torch

class ModelServer:
    def __init__(self, host, port, model):
        self.host = host
        self.port = port
        self.model = model
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def start(self):
        self.sock.bind((self.host, self.port))
        self.sock.listen(1)
        print(f"Model server listening on {self.host}:{self.port}...")

        while True:
            # Wait for a client to connect
            conn, addr = self.sock.accept()
            print(f"Connection from {addr} established.")

            # Pickle the state_dict of the model and send it to the client
            state_dict = self.model.state_dict()
            packed_state_dict = pickle.dumps(state_dict)
            conn.sendall(packed_state_dict)

            # Close the connection
            conn.close()
            print(f"Connection from {addr} closed.")

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

server = ModelServer('localhost', 8000, model)
server.start()
client = ModelClient('localhost', 8000)
client.connect()
state_dict = client.get_state_dict()
print(state_dict)
client.close()
