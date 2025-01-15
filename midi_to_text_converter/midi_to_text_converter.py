import os
import music21
import numpy as np

# Constants
VALTSEP = -1  # Separator value for numpy encoding
SAMPLE_FREQ = 4  # Sample frequency for resolution

def midi2txt(path):
    """Convert a MIDI file to text representation."""
    data = process_midi(path)
    result = []
    for array in data:
        val1, val2 = array[0], array[1]
        if val1 == VALTSEP:
            result.append(f"sepxx d{val2}")
        else:
            result.append(f"n{val1} d{val2}")
    result.insert(0, "START")
    return " ".join(result)

def process_midi(path):
    """Process MIDI file into numpy encoding."""
    if not os.path.exists(path):
        print(f"File {path} does not exist")
        return None

    # Load the MIDI file
    mf = music21.midi.MidiFile()
    mf.open(path)
    mf.read()
    mf.close()

    return midi2npenc(mf)

def midi2npenc(midi_file, skip_last_rest=True):
    """Convert MIDI to numpy encoding for text."""
    stream = file2stream(midi_file)  # Convert MIDI file to music21 stream
    chordarr = stream2chordarr(stream)  # Stream to chord array
    return chordarr2npenc(chordarr, skip_last_rest=skip_last_rest)

def file2stream(fp):
    """Convert file path to music21 stream."""
    if isinstance(fp, music21.midi.MidiFile):
        return music21.midi.translate.midiFileToStream(fp)
    return music21.converter.parse(fp)

def stream2chordarr(s, sample_freq=SAMPLE_FREQ, note_size=128, max_note_dur=32):
    """Convert music21 stream to numpy chord array."""
    highest_time = max(s.flat.getElementsByClass('Note').highestTime, s.flat.getElementsByClass('Chord').highestTime)
    maxTimeStep = round(highest_time * sample_freq) + 1
    score_arr = np.zeros((maxTimeStep, 1, note_size))

    def note_data(pitch, note):
        return pitch.midi, int(round(note.offset * sample_freq)), int(round(note.duration.quarterLength * sample_freq))

    for part in s.parts:
        notes = []
        for elem in part.flat:
            if isinstance(elem, music21.note.Note):
                notes.append(note_data(elem.pitch, elem))
            elif isinstance(elem, music21.chord.Chord):
                for p in elem.pitches:
                    notes.append(note_data(p, elem))

        for n in notes:
            pitch, offset, duration = n
            score_arr[offset, 0, pitch] = duration
    return score_arr

def chordarr2npenc(chordarr, skip_last_rest=True):
    """Convert chord array to numpy encoding."""
    result = []
    wait_count = 0
    for timestep in chordarr:
        flat_time = timestep2npenc(timestep)
        if len(flat_time) == 0:
            wait_count += 1
        else:
            if wait_count > 0:
                result.append([VALTSEP, wait_count])
            result.extend(flat_time)
            wait_count = 1
    if wait_count > 0 and not skip_last_rest:
        result.append([VALTSEP, wait_count])
    return np.array(result, dtype=int).reshape(-1, 2)

def timestep2npenc(timestep, note_range=(21, 108)):
    """Extract notes from a timestep."""
    notes = []
    for pitch, duration in enumerate(timestep[0]):
        if duration > 0:
            notes.append([pitch, int(duration)])
    return notes

def convert_midi_to_text(midi_path, output_dir):
    """Convert MIDI file to text file."""
    text = midi2txt(midi_path)
    name = os.path.basename(midi_path).replace(".mid", ".txt")
    output_path = os.path.join(output_dir, name)
    with open(output_path, 'w') as f:
        f.write(text)
    print(f"Converted {midi_path} to {output_path}")

if __name__ == "__main__":
    # Change these paths to your MIDI folder and output directory
    midi_dir = "../data/violin_dataset/midi_files"
    output_dir = "../converted_midi_to_text_files"

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all MIDI files in the directory
    for filename in os.listdir(midi_dir):
        if filename.endswith(".mid"):  # Process only MIDI files
            midi_path = os.path.join(midi_dir, filename)
            convert_midi_to_text(midi_path, output_dir)
