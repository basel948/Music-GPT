# this script will create a smaller version of the combined_dataset so that we can do a small test before really starting the main training.


input_file = "combined_dataset.txt"  # Path to the full dataset
output_file = "subset_dataset.txt"  # Path for the smaller dataset
num_sequences = 100  # Number of sequences to include in the subset

with open(input_file, "r") as infile:
    data = infile.readlines()

# Save a smaller subset of the data
with open(output_file, "w") as outfile:
    outfile.writelines(data[:num_sequences])

print(f"Subset created with {num_sequences} sequences.")
