import socket
import threading
import torch
import pickle 
import models
import utils as u

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
        self.lock_clients = threading.Lock()
        self.lock_aggregate = threading.Lock()
        self.round = 0

    def start(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, PORT))
            s.listen()
            print(f'[INFO] Server listening on {HOST}:{PORT}')
            while self.round < fl_rounds:
                conn, addr = s.accept()
                print(f'[INFO] Client {addr} connected')
                self.clients.append(conn)
                threading.Thread(target=self.handle_client, args=(conn,)).start()
                with self.lock_aggregate:
                    if len(self.clients_dict) >= num_clients:
                        self.server_aggregate()
                        self.round += 1

    def handle_client(self, conn):
        try:
            # Initialize client
            # TODO: Id client first, then decide if initialization is needed to save time
            u.send_pckld_bytes(conn, self.global_model.state_dict())
            print('[INFO] Global model sent to client')
            
            # --Receive and aggregate client models--
            client_model_dict = u.recv_pckld_bytes(conn)
            print('[INFO] Client model received & deserialized') 

            with self.lock_clients:
                self.clients_dict.append(client_model_dict)
                print('[INFO] Client model saved')
        except Exception as error:
            print('[ERROR] Error handling client')
            print(error)
        finally:
            conn.close()

    def server_aggregate(self):

        global_dict = self.global_model.state_dict() # Get a copy of the global model state_dict
        for key in global_dict.keys():
            # TODO: Fix indexes
            global_dict[key] = torch.stack([self.clients_dict[i][key].float() for i in range(len(self.clients))],0).mean(0)
        self.global_model.load_state_dict(global_dict)
        
if __name__ == '__main__':
    server = Server()
    server.start()
    print("[INFO] Finished training, testing")
