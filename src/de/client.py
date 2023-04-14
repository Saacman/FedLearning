import socket
import pickle
import torch

HOST = 'localhost' # The server's hostname or IP address
PORT = 65432 # The port used by the server

data = {'name': 'Alice', 'age': 25}

# with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#     s.connect((HOST, PORT))
#     s.sendall(pickle.dumps(data))
#     response_data = s.recv(1024)

# response = pickle.loads(response_data)
# print(response)


class Client:
    def __init__(self, host, port) -> None:
        self.host = host
        self.port = port
        self.sckt = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self):
        try:
            self.sckt.connect((self.host, self.port))
        except socket.error as e:
            print(str(e))
            
    def send(self, data):
        # Serialize data
        packed = pickle.dumps(data)

        # Send data to server
        self.sckt.sendall(packed)

        # Receive response from server
        #response = self.sckt.recv(1024)
        response = b''
        while True:
            print(len(response))
            chunk = self.sckt.recv(1024)
            if not chunk:
                break
            response += chunk

        # Deserialize response
        return pickle.loads(response)
    
    def close(self):
        self.sckt.close()
    
    def __exit__(self):
        self.close()

if __name__ == '__main__':
    # Create a client instance and connect to server
    client = Client(HOST, PORT)
    client.connect()

    # Send data to server and receive response
    data = {'message': 'Hello, server!'}
    response = client.send(data)

    # Print response
    print(response)

    # Load the state dict into a new PyTorch model
    new_model = torch.nn.Linear(10, 1)
    new_model.load_state_dict(response)

    # Close connection
    client.close()
