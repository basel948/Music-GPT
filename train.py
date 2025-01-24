import os
import argparse
import math
import torch
import sys
from torch.utils.data import DataLoader, random_split
from transformers import GPT2LMHeadModel, GPT2Config
from torch.utils.tensorboard import SummaryWriter
from dataset import TextDataset
import time
import datetime
from datetime import timedelta
from tqdm import tqdm  # Import tqdm

# Logging utilities
def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def get_lr(it, warmup_iters, lr_decay_iters, min_lr, max_lr):
    if it < warmup_iters:
        return max_lr * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def main(args):
    # Load config
    config_dict = {}
    exec(open(args.config).read(), config_dict)
    
    # Setup
    os.makedirs(config_dict['out_dir'], exist_ok=True)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(config_dict['out_dir'], "logs", run_id)
    writer = SummaryWriter(log_dir=log_dir)  # Modified line
    
    device = torch.device(config_dict.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Data
    dataset = TextDataset(config_dict['dataset'], config_dict['block_size'])
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], 
                                            generator=torch.Generator().manual_seed(42))
    
    # Model
    model_config = GPT2Config(
        vocab_size=dataset.vocab_size,
        n_positions=config_dict['block_size'],
        n_embd=config_dict['n_embd'],
        n_layer=config_dict['n_layer'],
        n_head=config_dict['n_head'],
        dropout=config_dict['dropout'],
        embd_pdrop=config_dict['dropout'],
        attn_pdrop=config_dict['dropout'],
        resid_pdrop=config_dict['dropout']
    )
    model = GPT2LMHeadModel(model_config).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), 
                                lr=config_dict['learning_rate'],
                                betas=(config_dict['beta1'], config_dict['beta2']),
                                weight_decay=config_dict['weight_decay'])
    
    # Dataloaders
    collate_fn = lambda batch: (torch.stack([x for x,y in batch]).to(device),
                                torch.stack([y for x,y in batch]).to(device))
    train_loader = DataLoader(train_dataset, batch_size=config_dict['batch_size'], 
                            shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config_dict['batch_size'],
                          collate_fn=collate_fn, drop_last=True)

    # Training state
    global_step = 0
    grad_accum_steps = config_dict['gradient_accumulation_steps']
    total_epochs = config_dict['max_iters']
    total_batches = 0
    # Main loop
    for epoch in range(total_epochs):
        model.train()
        epoch_start = time.time()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        optimizer.zero_grad()

        # Initialize tqdm progress bar
        pbar = tqdm(train_loader, 
                desc=f"Epoch {epoch+1}/{total_epochs}".ljust(20),
                bar_format="{l_bar}{bar:20}{r_bar}",
                dynamic_ncols=True,
                leave=True)
        for batch_idx, (x, y) in enumerate(pbar):
            total_batches += 1
            # LR scheduling
            if config_dict['decay_lr']:
                lr = get_lr(global_step, config_dict['warmup_iters'],
                           config_dict['lr_decay_iters'], config_dict['min_lr'],
                           config_dict['learning_rate'])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                lr = config_dict['learning_rate']

            # Forward pass
            outputs = model(x, labels=y)
            loss = outputs.loss / grad_accum_steps
            loss.backward()

            # Metrics
            with torch.no_grad():
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                correct = (preds == y).sum().item()
                total_correct += correct
                total_tokens += y.numel()
                total_loss += loss.item() * grad_accum_steps

            # Gradient accumulation
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                if config_dict['grad_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config_dict['grad_clip'])
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            # Update progress bar
            elapsed = time.time() - epoch_start
            samples_sec = (batch_idx + 1) * config_dict['batch_size'] / elapsed
            avg_loss = total_loss / (batch_idx + 1)
            avg_acc = total_correct / total_tokens
            pbar.set_postfix(ordered_dict={
                'loss': f"{avg_loss:.4f}",
                'acc': f"{avg_acc:.4f}",
                'lr': f"{lr:.1e}",
                'smp/s': f"{samples_sec:.0f}"
            }, refresh=False)
            writer.add_scalar('Loss/train', avg_loss, total_batches)
            writer.add_scalar('Accuracy/train', avg_acc, total_batches)
            writer.add_scalar('Learning Rate', lr, total_batches)
            # Validation
            if global_step % config_dict['eval_interval'] == 0 and (batch_idx + 1) % grad_accum_steps == 0:
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_tokens = 0
                with torch.no_grad():
                    for val_idx, (val_x, val_y) in enumerate(val_loader):
                        if val_idx >= config_dict['eval_iters']:
                            break
                        outputs = model(val_x, labels=val_y)
                        val_loss += outputs.loss.item()
                        logits = outputs.logits
                        preds = torch.argmax(logits, dim=-1)
                        val_correct += (preds == val_y).sum().item()
                        val_tokens += val_y.numel()
                avg_val_loss = val_loss / min(config_dict['eval_iters'], len(val_loader))
                val_acc = val_correct / val_tokens
                
                # TensorBoard logging
                writer.add_scalar('Loss/val', avg_val_loss, total_batches)
                writer.add_scalar('Accuracy/val', val_acc, total_batches)
                
                # Print validation results using tqdm to avoid breaking the progress bar
                pbar.write("\n" + "=" * 60)
                pbar.write(f"Validation @ step {global_step} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
                pbar.write("=" * 60 + "\n")
                model.train()

        # Close the progress bar for the current epoch
        pbar.close()

        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = total_loss / len(train_loader)
        avg_epoch_acc = total_correct / total_tokens
        
        print(f"\nEpoch {epoch+1} Summary | "
              f"Loss: {avg_epoch_loss:.4f} | Acc: {avg_epoch_acc:.4f} | "
              f"Time: {format_time(epoch_time)}")
        
        # Save checkpoint
        if (epoch + 1) % config_dict['save_interval'] == 0:
            ckpt_path = os.path.join(config_dict['out_dir'], f"ckpt_epoch{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GPT model with progress tracking")
    parser.add_argument("config", type=str, help="Path to config file")
    args = parser.parse_args()
    main(args)