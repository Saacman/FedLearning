import socket
import threading
import torch

HOST = '127.0.0.1'  # Server IP address
PORT = 65432  # Port to listen on

class Server:
    def __init__(self):
        self.global_model = None
        self.clients = []
        self.lock = threading.Lock()

    def start(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, PORT))
            s.listen()
            print(f'Server listening on {HOST}:{PORT}')
            while True:
                conn, addr = s.accept()
                print(f'Client {addr} connected')
                self.clients.append(conn)
                threading.Thread(target=self.handle_client, args=(conn,)).start()

    def handle_client(self, conn):
        try:
            # Send global model to client on first connection
            if self.global_model is None:
                self.lock.acquire()
                if self.global_model is None:
                    self.global_model = torch.rand(3, 3)  # initialize global model
                    print('Global model initialized')
                self.lock.release()
            global_model_bytes = torch.serialize(self.global_model)
            # Send size of global model first to handle truncated data
            size_bytes = len(global_model_bytes).to_bytes(4, byteorder='big')
            conn.sendall(size_bytes)
            conn.sendall(global_model_bytes)
            print('Global model sent to client')
            
            # Receive and aggregate client models
            size_bytes = conn.recv(4)
            size = int.from_bytes(size_bytes, byteorder='big')
            client_model_bytes = b''
            while len(client_model_bytes) < size:
                client_model_bytes += conn.recv(4096)
            client_model = torch.deserialize(client_model_bytes)
            self.lock.acquire()
            self.global_model += client_model
            self.lock.release()
            print('Client model aggregated')
        except:
            print('Error handling client')
        finally:
            conn.close()
if __name__ == '__main__':
    server = Server()
    server.start()
