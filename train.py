import os
import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Config
from trainer import Trainer
from dataset import TextDataset

# Argument parser for dynamic configuration
parser = argparse.ArgumentParser(description="Train a GPT model on a custom dataset")
parser.add_argument("config", type=str, help="Path to the training configuration file")

def main(args):
    # Load configuration
    config_path = args.config
    assert os.path.exists(config_path), f"Config file {config_path} does not exist."
    config = {}
    exec(open(config_path).read(), config)

    # Set up directories
    os.makedirs(config['out_dir'], exist_ok=True)
    
    # Check device
    device = torch.device(config.get('device', 'cpu'))
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = TextDataset(
        data_path=config['dataset'], 
        block_size=config['block_size']
    )

    # Initialize model
    print("Initializing model...")
    config = GPT2Config(
    vocab_size=dataset.vocab_size,
    n_positions=config['block_size'],
    n_ctx=config['block_size'],
    n_embd=config['n_embd'],
    n_layer=config['n_layer'],
    n_head=config['n_head']
    )

    model = GPT2LMHeadModel(config).to(device)

    # Set up optimizer and trainer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        betas=(config['beta1'], config['beta2']),
        weight_decay=config['weight_decay']
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataset=dataset.get_split('train', 0.9),
        val_dataset=dataset.get_split('val', 0.1),
        config=config
    )

    # Start training
    print("Starting training...")
    trainer.train()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
