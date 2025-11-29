import mido
import librosa
import numpy as np

def map_event_to_midi(event_analysis, event_label):
    """
    Maps the analysis of an audio event to a MIDI note.

    Args:
        event_analysis (dict): A dictionary containing 'pitch', 'duration', and 'volume'.
        event_label (str): The label of the event (e.g., 'Tonal/Whistle').

    Returns:
        mido.Message: A MIDI message representing the event. Returns None if pitch is invalid.
    """
    pitch_hz = event_analysis.get('pitch')
    if pitch_hz is None or pitch_hz <= 0:
        return None

    # Convert frequency (Hz) to MIDI note number
    midi_note = int(round(librosa.hz_to_midi(pitch_hz)))

    # Map volume to MIDI velocity (0-127)
    velocity = int(np.clip(event_analysis['volume'], 0, 127))

    # For now, we'll create a simple note_on message.
    # The duration will be handled when we create the MIDI file.
    return mido.Message('note_on', note=midi_note, velocity=velocity, time=0)


def save_midi_file(midi_messages, output_path, ticks_per_beat=480):
    """
    Saves a list of MIDI messages to a .mid file.

    Args:
        midi_messages (list): A list of mido.Message objects.
        output_path (str): The path to save the MIDI file.
        ticks_per_beat (int): The number of ticks per beat (resolution).
    """
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    for msg in midi_messages:
        # Here, we'd ideally calculate the time delta between events.
        # For simplicity, we'll add a small, fixed delay.
        time_delta = int(mido.second2tick(0.1, ticks_per_beat, mido.bpm2tempo(120)))
        track.append(msg.copy(time=time_delta))

        # Add a corresponding note_off message
        if msg.type == 'note_on':
            note_off_time = int(mido.second2tick(0.4, ticks_per_beat, mido.bpm2tempo(120)))
            track.append(mido.Message('note_off', note=msg.note, velocity=msg.velocity, time=note_off_time))


    # Ensure the output directory exists
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mid.save(output_path)
    print(f"MIDI file saved to {output_path}")


if __name__ == '__main__':
    # Example Usage
    from src.analysis import analyze_event
    import numpy as np
    import os

    # 1. Create a dummy audio segment
    sr = 44100
    duration = 0.5  # seconds
    frequency = 261.63  # Middle C
    t = np.linspace(0., duration, int(sr * duration))
    amplitude = 0.8
    audio_segment = amplitude * np.sin(2. * np.pi * frequency * t)

    # 2. Analyze the segment
    analysis_results = analyze_event(audio_segment, sr=sr)
    print(f"Analysis Results: {analysis_results}")

    # 3. Map to a MIDI message
    event_label = 'Tonal/Whistle'
    midi_message = map_event_to_midi(analysis_results, event_label)

    if midi_message:
        print(f"Generated MIDI Message: {midi_message}")

        # 4. Save to a MIDI file
        # create output folder if it doesn't exist
        if not os.path.exists('output'):
            os.makedirs('output')
        output_midi_path = 'output/example.mid'
        save_midi_file([midi_message, midi_message, midi_message], output_midi_path)
    else:
        print("Failed to generate MIDI message.")
