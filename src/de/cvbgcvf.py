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


import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 2),
    nn.Sigmoid()
)

server = ModelServer('localhost', 8000, model)
server.start()
