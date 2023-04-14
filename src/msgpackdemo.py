import torch
import msgpack

# Create a PyTorch model and get its state dict
model = torch.nn.Linear(10, 1)
state_dict = model.state_dict()
print(type(state_dict))
# Serialize the state dict using MessagePack
packed = msgpack.packb(state_dict, use_bin_type=True)

# Deserialize the state dict using MessagePack
unpacked = msgpack.unpackb(packed, raw=False)

# Load the state dict into a new PyTorch model
new_model = torch.nn.Linear(10, 1)
new_model.load_state_dict(unpacked)