import socket
import pickle
import torch
import select 

HOST = 'localhost' # The server's hostname or IP address
PORT = 65433 # The port used by the server

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
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #self.sock.setblocking(0)

    def connect(self):
        try:
            self.sock.connect((self.host, self.port))
        except socket.error as e:
            print(str(e), "hola")

    def send(self, data):
        #self.sock.settimeout(2)
        # Serialize data
        packed = pickle.dumps(data)

        # Send data to server
        self.sock.sendall(packed)

        # Receive response from server
        response = []
        chunk_len = 1024
        while True:
            # chunk = self.sock.recv(chunk_len)
            # response.append(chunk)
            # if len(chunk) < chunk_len:
            #     break
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
    new_model = torch.nn.Linear(10, 1)
    new_model.load_state_dict(response)

    # Close connection
    client.close()
