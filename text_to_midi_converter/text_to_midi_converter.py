import os
import music21

# Constants
VALTSEP = "sepxx"
NOTE_PREFIX = "n"
DURATION_PREFIX = "d"

import os
import music21

# Constants
VALTSEP = "sepxx"
NOTE_PREFIX = "n"
DURATION_PREFIX = "d"

def text_to_midi(text_file, output_midi_path, tempo=120):
    """Convert text representation to a MIDI file."""
    # Read the text file
    with open(text_file, 'r') as f:
        content = f.read().strip()
    
    # Split into tokens
    tokens = content.split()
    
    # Validate format
    if not tokens or tokens[0] != "START":
        raise ValueError("Invalid text format. Must start with 'START'.")
    
    # Parse tokens into notes and durations
    stream = music21.stream.Stream()
    stream.append(music21.instrument.Violin())  # Set the instrument to Violin
    
    # Set tempo
    tempo_mark = music21.tempo.MetronomeMark(number=tempo)
    stream.append(tempo_mark)
    
    pitch = None
    for token in tokens[1:]:
        if token.startswith(NOTE_PREFIX):
            pitch = int(token[1:])
        elif token.startswith(DURATION_PREFIX) and pitch is not None:
            # Scale duration to match the original tempo
            duration = float(token[1:]) / 4  # Assuming durations are scaled by 4
            note = music21.note.Note(pitch)
            note.quarterLength = duration
            stream.append(note)
            pitch = None  # Reset pitch after creating a note
        elif token == VALTSEP:
            stream.append(music21.note.Rest(quarterLength=0.25))  # Default quarter rest

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_midi_path), exist_ok=True)

    # Write the MIDI file
    stream.write('midi', fp=output_midi_path)
    print(f"Converted {text_file} to {output_midi_path}")


# Example usage
if __name__ == "__main__":
    text_file = "../converted_midi_to_text_files/Kayser_Op20-01_AlexandrosIakovou_O105paQOHCE-0004-0064.txt"
    output_midi_path = "../converted_text_to_midi_files/Kayser_Op20-01_AlexandrosIakovou_O105paQOHCE-0004-0064.mid"
    text_to_midi(text_file, output_midi_path)
