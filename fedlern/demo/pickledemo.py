import torch
import pickle

# Create a PyTorch model and get its state dict
model = torch.nn.Linear(10, 1)
state_dict = model.state_dict()

# Serialize the state dict using pickle
packed = pickle.dumps(state_dict)

print(len(packed))

# Deserialize the state dict using pickle
unpacked = pickle.loads(packed)

# Load the state dict into a new PyTorch model
new_model = torch.nn.Linear(10, 1)
new_model.load_state_dict(unpacked)