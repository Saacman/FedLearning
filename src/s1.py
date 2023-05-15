import socket
import threading
import torch
import pickle 
import models

HOST = '127.0.0.1'  # Server IP address
PORT = 65432  # Port to listen on

fl_rounds = 25
num_clients = 10
sample_size = int(.3 * num_clients) # Use 30% of available clients
net_parameters = [ 28 * 28, # input
                512, 256, 128, 64,
                10 ] #output
class Server:
    def __init__(self):
        self.global_model = models.MLP(net_parameters)  # initialize local model
        self.clients = []
        self.clients_dict = []
        self.lock = threading.Lock()
        self.round = 0

    def start(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, PORT))
            s.listen()
            print(f'Server listening on {HOST}:{PORT}')
            while self.round < fl_rounds:
                conn, addr = s.accept()
                print(f'Client {addr} connected')
                self.clients.append(conn)
                threading.Thread(target=self.handle_client, args=(conn,)).start()
                # TODO: better check if this is correct with threads
                if len(self.clients_dict) >= num_clients:
                    self.server_aggregate()
                    self.round += 1

    def handle_client(self, conn):
        try:
            global_model_bytes = pickle.dumps(self.global_model.state_dict())
            # Send size of global model first to handle truncated data
            size_bytes = len(global_model_bytes).to_bytes(4, byteorder='big')
            conn.sendall(size_bytes)
            conn.sendall(global_model_bytes)
            print('Global model sent to client')
            
            # Receive and aggregate client models
            size_bytes = b''
            while len(size_bytes) < 4:
                size_bytes += conn.recv(4 - len(size_bytes))
            size = int.from_bytes(size_bytes, byteorder='big')

            client_model_bytes = b''
            while len(client_model_bytes) < size:
                client_model_bytes += conn.recv(4096)

            client_model_dict = pickle.loads(client_model_bytes)
            print('Client model received & deserialized') 

            self.lock.acquire()
            self.clients_dict.append(client_model_dict)
            self.lock.release()
            print('Client model saved')
        except Exception as error:
            print('Error handling client')
            print(error)
        finally:
            conn.close()

    def server_aggregate(self):

        global_dict = self.global_model.state_dict() # Get a copy of the global model state_dict
        for key in global_dict.keys():
            global_dict[key] = torch.stack([self.clients_dict[i][key].float() for i in range(len(self.clients))],0).mean(0)
        self.global_model.load_state_dict(global_dict)
        
if __name__ == '__main__':
    server = Server()
    server.start()
    print("Finished training, testing")
