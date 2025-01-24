import os
import argparse
import math
import torch
from torch.utils.data import DataLoader, random_split
from transformers import GPT2LMHeadModel, GPT2Config
from torch.utils.tensorboard import SummaryWriter
from dataset import TextDataset
import time

def get_lr(it, warmup_iters, lr_decay_iters, min_lr, max_lr):
    if it < warmup_iters:
        return max_lr * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def main(args):
    config_path = args.config
    assert os.path.exists(config_path), f"Config file {config_path} does not exist."
    config_dict = {}
    exec(open(config_path).read(), config_dict)

    os.makedirs(config_dict['out_dir'], exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(config_dict['out_dir'], "logs"))

    device = torch.device(config_dict.get('device', 'cpu'))
    print(f"Using device: {device}")

    dataset = TextDataset(
        data_path=config_dict['dataset'], 
        block_size=config_dict['block_size']
    )

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

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

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config_dict['learning_rate'],
        betas=(config_dict['beta1'], config_dict['beta2']),
        weight_decay=config_dict['weight_decay']
    )

    def collate_fn(batch):
        x = torch.stack([item[0] for item in batch])
        y = torch.stack([item[1] for item in batch])
        return x.to(device), y.to(device)

    train_loader = DataLoader(train_dataset, batch_size=config_dict['batch_size'], shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config_dict['batch_size'], collate_fn=collate_fn, drop_last=True)

    global_step = 0
    grad_accum_steps = config_dict['gradient_accumulation_steps']
    
    for epoch in range(config_dict['max_iters']):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        accum_loss = 0.0
        accum_correct = 0
        accum_tokens = 0
        batch_start_time = time.time()
        optimizer.zero_grad()

        for batch_idx, (x, y) in enumerate(train_loader):
            if config_dict['decay_lr']:
                lr = get_lr(global_step, config_dict['warmup_iters'], 
                           config_dict['lr_decay_iters'], config_dict['min_lr'],
                           config_dict['learning_rate'])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            
            outputs = model(x, labels=y)
            loss = outputs.loss
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            correct = (preds == y).sum().item()
            total_correct += correct
            accum_correct += correct
            total_tokens += y.numel()
            accum_tokens += y.numel()

            loss = loss / grad_accum_steps
            loss.backward()
            accum_loss += loss.item()
            total_loss += loss.item()

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                if config_dict['grad_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config_dict['grad_clip'])
                optimizer.step()
                optimizer.zero_grad()

                train_loss = accum_loss * grad_accum_steps
                train_acc = accum_correct / accum_tokens
                train_ppl = math.exp(train_loss)

                writer.add_scalar('Loss/train', train_loss, global_step)
                writer.add_scalar('Accuracy/train', train_acc, global_step)
                writer.add_scalar('Perplexity/train', train_ppl, global_step)
                writer.add_scalar('Learning Rate', lr, global_step)

                accum_loss = 0.0
                accum_correct = 0
                accum_tokens = 0
                global_step += 1

            if (batch_idx + 1) % config_dict['log_interval'] == 0:
                avg_loss = total_loss / (batch_idx + 1) * grad_accum_steps
                avg_acc = total_correct / total_tokens
                samples_per_sec = config_dict['batch_size'] * grad_accum_steps * config_dict['log_interval'] / (time.time() - batch_start_time)
                
                print(f"Epoch {epoch+1} | Batch {batch_idx+1} | "
                      f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | "
                      f"LR: {lr if config_dict['decay_lr'] else config_dict['learning_rate']:.2e} | "
                      f"Samples/s: {samples_per_sec:.1f}")
                batch_start_time = time.time()

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
                val_ppl = math.exp(avg_val_loss)

                writer.add_scalar('Loss/val', avg_val_loss, global_step)
                writer.add_scalar('Accuracy/val', val_acc, global_step)
                writer.add_scalar('Perplexity/val', val_ppl, global_step)

                print(f"\n{'='*80}\nEVALUATION | Step {global_step} | "
                      f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                      f"Val PPL: {val_ppl:.2f}\n{'='*80}\n")
                model.train()

        if (epoch + 1) % config_dict['save_interval'] == 0:
            checkpoint_path = os.path.join(config_dict['out_dir'], f"checkpoint_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GPT model on a custom dataset")
    parser.add_argument("config", type=str, help="Path to the training configuration file")
    args = parser.parse_args()
    main(args)