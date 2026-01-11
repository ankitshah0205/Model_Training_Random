script_content = """
import torch
from transformers import GPT2LMHeadModel, GPT2Config
from torch.utils.data import Dataset, DataLoader
import deepspeed

class DummyDataset(Dataset):
    def __len__(self): return 100
    def __getitem__(self, idx):
        return torch.randint(0, 50257, (128,))

model = GPT2LMHeadModel(GPT2Config(n_layer=4)) # Small model for speed
dataset = DummyDataset()

ds_config = {
    "train_micro_batch_size_per_gpu": 1,
    "optimizer": {"type": "Adam", "params": {"lr": 1e-4}},
    "fp16": {"enabled": True},
    "zero_optimization": {"stage": 2}
}

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config
)

for batch in DataLoader(dataset, batch_size=1):
    inputs = batch.to(model_engine.local_rank)
    outputs = model_engine(inputs, labels=inputs)
    loss = outputs.loss
    model_engine.backward(loss)
    model_engine.step()
    print(f"Loss: {loss.item()}")
"""
with open("train_ds.py", "w") as f:
    f.write(script_content)
