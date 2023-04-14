import socket
import pickle
import torch
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
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self):
        try:
            self.sock.connect((self.host, self.port))
        except socket.error as e:
            print(str(e))

    def send(self, data):
        # Serialize data
        packed = pickle.dumps(data)

        # Send data to server
        self.sock.sendall(packed)

        # Receive response from server
        response = []
        chunk_len = 1024
        while True:
            chunk = self.sock.recv(chunk_len)
            if not chunk:
                break
            response.append(chunk)


        # Deserialize response
        return pickle.loads(b"".join(response))
    
    def close(self):
        self.sock.close()
    
    def __exit__(self):
        self.close()

if __name__ == '__main__':
    # Create a client instance and connect to server
    client = Client(HOST, PORT)
    client.connect()

    # Send data to server and receive response
    data = {'status': 'uinit'}
    response = client.send(data)

    # Print response
    print(response)

    # Load the state dict into a new PyTorch model
    new_model = MLP(net_parameters)
    new_model.load_state_dict(response)

    # Close connection
    client.close()
