import os
import argparse
from src.normalization import load_audio, normalize_audio, high_pass_filter
from src.classification import classify_audio_events
from src.analysis import analyze_event
from src.midi_mapping import map_event_to_midi, save_midi_file

def main(input_file, output_file):
    """
    Main function to run the Fart-to-MIDI pipeline.
    """
    print(f"Processing audio file: {input_file}")

    # 1. Audio Acquisition & Normalization
    audio, sr = load_audio(input_file)
    if audio is None:
        return

    normalized_audio = normalize_audio(audio)
    filtered_audio = high_pass_filter(normalized_audio)
    print("Audio normalized and filtered.")

    # 2. Fart Identification and Labeling (Placeholder)
    events = classify_audio_events(filtered_audio, sr)
    print(f"Detected {len(events)} events.")

    # 3. Transient Analysis and Segmentation & 4. MIDI Mapping
    midi_messages = []
    for event in events:
        start_sample = int(event['start_time'] * sr)
        end_sample = int(event['end_time'] * sr)
        audio_segment = filtered_audio[start_sample:end_sample]

        if len(audio_segment) > 0:
            analysis_results = analyze_event(audio_segment, sr)
            midi_message = map_event_to_midi(analysis_results, event['label'])
            if midi_message:
                midi_messages.append(midi_message)

    print(f"Generated {len(midi_messages)} MIDI messages.")

    # 5. MIDI Output Generation
    if midi_messages:
        save_midi_file(midi_messages, output_file)
    else:
        print("No MIDI messages were generated.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fart-to-MIDI Audio Pipeline')
    parser.add_argument('input_file', type=str, help='Path to the input audio file.')
    parser.add_argument('output_file', type=str, help='Path to the output MIDI file.')
    args = parser.parse_args()

    # Check if the input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found at {args.input_file}")
        # Optionally, create a dummy file for demonstration
        import numpy as np
        import soundfile as sf

        # Ensure the data directory exists
        os.makedirs(os.path.dirname(args.input_file), exist_ok=True)

        print(f"Creating a dummy audio file at: {args.input_file}")
        sr = 44100
        duration = 5
        frequency = 440
        t = np.linspace(0., duration, int(sr * duration))
        amplitude = np.iinfo(np.int16).max * 0.5
        data = amplitude * np.sin(2. * np.pi * frequency * t)
        sf.write(args.input_file, data.astype(np.int16), sr)
        print("Dummy file created. Please run the script again.")
    else:
        main(args.input_file, args.output_file)
