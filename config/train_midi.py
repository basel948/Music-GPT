# train_config.py
out_dir = 'out-midi'
eval_interval = 100
eval_iters = 50
log_interval = 1
save_interval = 10

# Data settings
dataset = 'combined_dataset.txt'
gradient_accumulation_steps = 4  # Allows larger effective batch size
batch_size = 64  # Reduce if OOM occurs
block_size = 256  # Balance context length vs memory

# Model architecture (optimized for small memory)
n_layer = 3  # Reduced layers
n_head = 4   # Fewer attention heads
n_embd = 128 # Smaller embedding dimension
dropout = 0.2 # Increased dropout for regularization

# Training parameters
learning_rate = 3e-4
max_iters = 1000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule
decay_lr = True
warmup_iters = 100
lr_decay_iters = 800
min_lr = 1e-5

# System
device = 'cuda'