# generate.py
import torch
from dataset import TextDataset
from transformers import GPT2LMHeadModel, GPT2Config
from text_to_midi_converter.text_to_midi_converter import text_to_midi
import os

# Load the same config as used during training
from config.train_midi import *

# Define output directory
output_dir = "generated_outputs"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Load dataset to get vocab mappings
dataset_path = "combined_dataset.txt"
block_size = 256  # Ensure this matches your training config
dataset = TextDataset(dataset_path, block_size)
stoi = dataset.stoi
itos = dataset.itos

# Load model
checkpoint_path = "out-midi/ckpt_epoch35.pt"
checkpoint = torch.load(checkpoint_path)

model_config = GPT2Config(
    vocab_size=dataset.vocab_size,
    n_positions=block_size,
    n_embd=n_embd,
    n_layer=n_layer,
    n_head=n_head,
    dropout=dropout
)
model = GPT2LMHeadModel(model_config)
model.load_state_dict(checkpoint['model_state'])
model.eval()

def generate_sequence(model, stoi, itos, block_size, start_token="START", max_length=500):
    input_ids = [stoi[start_token]]
    for _ in range(max_length):
        inputs = input_ids[-block_size:]
        inputs_tensor = torch.tensor(inputs).unsqueeze(0)
        with torch.no_grad():
            outputs = model(inputs_tensor)
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
        input_ids.append(next_token)
        if itos[next_token] == "END":
            break
    return ' '.join([itos[i] for i in input_ids])

# Generate and save MIDI
generated_text = generate_sequence(model, stoi, itos, block_size)
output_text_path = os.path.join(output_dir, "generated_sample.txt")
output_midi_path = os.path.join(output_dir, "generated_sample.mid")

with open(output_text_path, "w") as f:
    f.write(generated_text)
text_to_midi(output_text_path, output_midi_path)

print(f"Generated files saved in: {output_dir}")
