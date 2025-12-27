import torch, os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from models.rnn import RecurrentModel

model = RecurrentModel(model_type="GRU")

dummy = torch.randn(2, 300, 512)   # [batch, time, features]
out = model(dummy)

print(out.shape)   # expect: [2, 300, 41]
