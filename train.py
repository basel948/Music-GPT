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
from datetime import timedelta  # Add this line
# Logging utilities
def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def progress_bar(current, total, length=30):
    filled = int(round(length * current / total))
    bar = '■' * filled + ' ' * (length - filled)
    return f"[{bar}] {current}/{total}"

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
    writer = SummaryWriter(log_dir=os.path.join(config_dict['out_dir'], "logs"))
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
    
    # Main loop
    for epoch in range(total_epochs):
        model.train()
        epoch_start = time.time()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        optimizer.zero_grad()

        for batch_idx, (x, y) in enumerate(train_loader):
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

            # Progress bar
            elapsed = time.time() - epoch_start
            samples_sec = (batch_idx + 1) * config_dict['batch_size'] / elapsed
            avg_loss = total_loss / (batch_idx + 1)
            avg_acc = total_correct / total_tokens
            
            if batch_idx % 10 == 0 or batch_idx == len(train_loader)-1:
                sys.stdout.write('\r\033[K')
                sys.stdout.write(
                    f"Epoch {epoch+1}/{total_epochs} "
                    f"{progress_bar(batch_idx+1, len(train_loader))} | "
                    f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | "
                    f"LR: {lr:.2e} | {samples_sec:.0f} samples/s | "
                    f"Elapsed: {format_time(elapsed)}"
                )
                sys.stdout.flush()

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
                writer.add_scalar('Loss/train', avg_loss, global_step)
                writer.add_scalar('Accuracy/train', avg_acc, global_step)
                writer.add_scalar('Loss/val', avg_val_loss, global_step)
                writer.add_scalar('Accuracy/val', val_acc, global_step)
                
                # Print validation results
                sys.stdout.write(f"\n{'='*60}\n")
                sys.stdout.write(f"Validation @ step {global_step} | "
                               f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}\n")
                sys.stdout.write(f"{'='*60}\n")
                model.train()

        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = total_loss / len(train_loader)
        avg_epoch_acc = total_correct / total_tokens
        
        sys.stdout.write(f"\nEpoch {epoch+1} Summary | "
                       f"Loss: {avg_epoch_loss:.4f} | Acc: {avg_epoch_acc:.4f} | "
                       f"Time: {format_time(epoch_time)}\n")
        
        # Save checkpoint
        if (epoch + 1) % config_dict['save_interval'] == 0:
            ckpt_path = os.path.join(config_dict['out_dir'], f"ckpt_epoch{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, ckpt_path)
            sys.stdout.write(f"Saved checkpoint to {ckpt_path}\n")

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GPT model with progress tracking")
    parser.add_argument("config", type=str, help="Path to config file")
    args = parser.parse_args()
    main(args)