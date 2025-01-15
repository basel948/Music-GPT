import os

# this script add the token END for each text file, and after that it combines all the text files into one large file.

# Define paths
input_folder = "converted_midi_to_text_files"  # Folder containing your original text files
combined_file = "combined_dataset.txt"  # Final combined file

# Step 1: Add END token directly to the original files
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        input_path = os.path.join(input_folder, filename)

        # Read the original file content
        with open(input_path, "r") as infile:
            content = infile.read().strip()

        # Append the END token
        updated_content = f"{content} END"

        # Overwrite the original file with the updated content
        with open(input_path, "w") as outfile:
            outfile.write(updated_content)

print(f"All original files updated with 'END' token.")

# Step 2: Combine all updated files into one file
with open(combined_file, "w") as combined:
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_folder, filename)

            # Read the updated file content
            with open(file_path, "r") as infile:
                content = infile.read().strip()

            # Write the content to the combined file
            combined.write(f"{content}\n")  # No SONG_END added

print(f"All text files combined into: {combined_file}")
