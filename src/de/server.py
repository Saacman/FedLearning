import socket
import threading
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from model import MLP

HOST = 'localhost' # The server's hostname or IP address
PORT = 65432 # The port used by the server

class Server:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.host, self.port))
        self.sock.listen(1)
        print(f"Listening on {self.host}:{self.port}")

        self.model = torch.nn.Linear(10, 1)
        self.state_dict = self.model.state_dict()

    def handle_client(self, conn, addr):
        with conn:
            i = 0
            print('Connected by', addr)
            while True:
                print(i)
                i += 1
                data = conn.recv(1024)
                if not data:
                    break
                obj = pickle.loads(data)
                # Do something with obj
                print(obj)

                state_dict = self.model.state_dict()
                print(state_dict)
                response = {'status': 'OK'}
                packed = pickle.dumps(state_dict)
                print(len(packed))
                a = conn.sendall(packed)
                print(a)

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
    # Create a server instance and start it
    server = Server(HOST, PORT)
    server.start()
