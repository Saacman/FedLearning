import socket
import threading
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from model import MLP

HOST = 'localhost' # The server's hostname or IP address
PORT = 65433 # The port used by the server

epoch = 5
comm_cycles = 25
num_clients = 10
sample_size = int(.3 * num_clients) # Use 30% of available clients
net_parameters = [ 28 * 28, # input
                512, 256, 128, 64,
                10 ] #output

class Server:
    def __init__(self, host, port, model):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.host, self.port))
        self.sock.listen(1)
        print(f"Listening on {self.host}:{self.port}")

        self.model = model


    def handle_client(self, conn, addr):
        print('Connected by', addr)
        with conn:
            #while True:
            data = conn.recv(1024)
            # if not data:
            #     break
            obj = pickle.loads(data)
            
            # If client is uunit, send the global model
            if obj['status'] == 'uinit':
                packed = pickle.dumps(self.model.state_dict())
            elif obj['status'] == 'submt':
                packed = pickle.dumps({'status': 'ready'})
            conn.sendall(packed)

        print(f"Connection closed")

        

    def start(self):
        while True:
            # Wait for a new client connection
            conn, addr = self.sock.accept()

            # Start a new thread to handle the client
            client_thread = threading.Thread(target=self.handle_client, args=(conn, addr))
            client_thread.start()

    def stop(self):
        self.sock.close()
    
    def __exit__(self):
        self.stop()

if __name__ == '__main__':
    # Global Model instantiation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global_model = MLP(net_parameters)
    global_model.to(device)

    # Create a server instance and start it
    server = Server(HOST, PORT, global_model)
    server.start()
