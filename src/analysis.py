import librosa
import numpy as np

def analyze_event(audio_segment, sr=44100):
    """
    Analyzes an audio segment to extract musical parameters.

    Args:
        audio_segment (np.ndarray): The audio data for the event.
        sr (int): The sample rate.

    Returns:
        dict: A dictionary containing the extracted parameters:
              'pitch', 'duration', and 'volume'.
    """
    # Duration calculation
    duration = librosa.get_duration(y=audio_segment, sr=sr)

    # Volume (RMS energy)
    rms = np.sqrt(np.mean(audio_segment**2))
    volume = rms * 100  # Scale for MIDI velocity

    # Pitch estimation (using Yin algorithm)
    f0, voiced_flag, voiced_probs = librosa.pyin(audio_segment, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    # Get the most prominent pitch
    pitch = np.nanmedian(f0[voiced_flag]) if np.any(voiced_flag) else 60.0  # Default to middle C

    return {
        'pitch': pitch,
        'duration': duration,
        'volume': volume
    }

if __name__ == '__main__':
    # Example usage:
    from src.classification import classify_audio_events

    # Create a dummy audio signal for testing
    sr = 44100
    duration = 5
    frequency = 440  # A4
    t = np.linspace(0., duration, int(sr * duration))
    amplitude = 0.5
    audio = amplitude * np.sin(2. * np.pi * frequency * t)

    # Get mock events from the classifier
    events = classify_audio_events(audio, sr)

    print("Analyzing detected events:")
    for event in events:
        start_sample = int(event['start_time'] * sr)
        end_sample = int(event['end_time'] * sr)
        audio_segment = audio[start_sample:end_sample]

        if len(audio_segment) > 0:
            analysis_results = analyze_event(audio_segment, sr)
            print(f"- Event: {event['label']}")
            print(f"  - Pitch: {analysis_results['pitch']:.2f} Hz")
            print(f"  - Duration: {analysis_results['duration']:.2f} s")
            print(f"  - Volume: {analysis_results['volume']:.2f}")
