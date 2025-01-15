# train_midi.py

# Training configuration for MIDI dataset
out_dir = 'out-midi'  # Directory to save model checkpoints
eval_interval = 100  # Evaluate the model every 100 iterations
eval_iters = 20  # Number of iterations to use for evaluation
log_interval = 10  # Log training progress every 10 iterations
# Data settings
dataset = 'subset_dataset.txt'  # Name of your dataset
gradient_accumulation_steps = 8  # Accumulate gradients over 8 steps
batch_size = 4  # Number of samples per batch
block_size = 64  # Context window size

# Model hyperparameters
n_layer = 4  # Number of transformer layers
n_head = 4  # Number of attention heads
n_embd = 256  # Embedding dimension
dropout = 0.1  # Dropout rate

# Optimization settings
learning_rate = 3e-4  # Initial learning rate
max_iters = 100  # Total number of training iterations
weight_decay = 1e-2  # Weight decay for regularization
beta1 = 0.9  # Adam optimizer beta1 parameter
beta2 = 0.95  # Adam optimizer beta2 parameter
grad_clip = 1.0  # Gradient clipping threshold

# Learning rate decay
decay_lr = True  # Enable learning rate decay
warmup_iters = 100  # Number of warmup iterations
lr_decay_iters = 500  # Iterations over which to decay the learning rate
min_lr = 1e-5  # Minimum learning rate

# Device settings
device = 'cuda'  # Use 'cuda' for GPU training or 'cpu' for CPU training
