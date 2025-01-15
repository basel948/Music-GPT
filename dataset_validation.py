import re

# Define the path to your dataset file
dataset_file = "combined_dataset.txt"

# Define expected tokens in the vocabulary
vocabulary = {"START", "END", "sepxx"}
durations = {f"d{i}" for i in range(1, 17)}  # d1 to d16
notes = {f"n{i}" for i in range(60, 85)}    # n60 to n84 (example range)
vocabulary.update(durations)
vocabulary.update(notes)

# Parameters for validation
max_sequence_length = 1024  # Example context window size of your model

# Initialize validation flags and statistics
sequence_lengths = []
errors = []

# Validation functions
def validate_tokens(sequence):
    """Validate that all tokens in the sequence are in the vocabulary."""
    tokens = sequence.split()
    for token in tokens:
        if token not in vocabulary:
            return f"Invalid token: {token}"
    return None

def validate_sequence_structure(sequence):
    """Validate that a sequence starts with START and ends with END."""
    if not sequence.startswith("START"):
        return "Sequence does not start with START."
    if not sequence.endswith("END"):
        return "Sequence does not end with END."
    return None

def validate_formatting(sequence):
    """Ensure that the sequence contains consistent formatting."""
    # Check for missing separators (e.g., "sepxx" between tokens)
    if "  " in sequence or "\t" in sequence:
        return "Sequence contains unexpected whitespace or tabs."
    if "sepxx" not in sequence:
        return "Missing 'sepxx' token in sequence."
    return None

# Process the dataset
with open(dataset_file, "r") as file:
    data = file.read().strip().split("\n")

for i, sequence in enumerate(data):
    sequence_length = len(sequence.split())
    sequence_lengths.append(sequence_length)

    # Check sequence length
    if sequence_length > max_sequence_length:
        errors.append(f"Sequence {i+1}: Exceeds maximum length ({sequence_length} tokens).")

    # Validate tokens
    token_error = validate_tokens(sequence)
    if token_error:
        errors.append(f"Sequence {i+1}: {token_error}")

    # Validate structure
    structure_error = validate_sequence_structure(sequence)
    if structure_error:
        errors.append(f"Sequence {i+1}: {structure_error}")

    # Validate formatting
    formatting_error = validate_formatting(sequence)
    if formatting_error:
        errors.append(f"Sequence {i+1}: {formatting_error}")

# Output results
print(f"Total sequences: {len(sequence_lengths)}")
print(f"Max sequence length: {max(sequence_lengths)}")
print(f"Min sequence length: {min(sequence_lengths)}")
print(f"Average sequence length: {sum(sequence_lengths) / len(sequence_lengths):.2f}")

if errors:
    print("\nErrors found:")
    for error in errors:
        print(error)
else:
    print("\nDataset validation passed with no errors!")
