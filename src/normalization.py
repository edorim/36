import librosa
import numpy as np
from scipy.signal import butter, lfilter

def load_audio(file_path, sr=44100):
    """Loads an audio file.

    Args:
        file_path (str): The path to the audio file.
        sr (int, optional): The target sample rate. Defaults to 44100.

    Returns:
        np.ndarray: The audio data.
        int: The sample rate.
    """
    try:
        audio, sr = librosa.load(file_path, sr=sr)
        return audio, sr
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None, None

def normalize_audio(audio, target_dbfs=-6.0):
    """Normalizes the audio to a target dBFS level.

    Args:
        audio (np.ndarray): The audio data.
        target_dbfs (float, optional): The target dBFS level. Defaults to -6.0.

    Returns:
        np.ndarray: The normalized audio data.
    """
    rms = np.sqrt(np.mean(audio**2))
    if rms == 0:
        return audio # Avoid division by zero for silence
    target_rms = 10**(target_dbfs / 20.0)
    return audio * (target_rms / rms)

def high_pass_filter(audio, cutoff=100, sr=44100, order=5):
    """Applies a high-pass filter to the audio.

    Args:
        audio (np.ndarray): The audio data.
        cutoff (int, optional): The cutoff frequency in Hz. Defaults to 100.
        sr (int, optional): The sample rate. Defaults to 44100.
        order (int, optional): The filter order. Defaults to 5.

    Returns:
        np.ndarray: The filtered audio data.
    """
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return lfilter(b, a, audio)

if __name__ == '__main__':
    # Example usage
    # Create a dummy audio signal for testing
    sr = 44100
    duration = 5
    frequency = 440
    t = np.linspace(0., duration, int(sr * duration))
    amplitude = 0.5 # Use float amplitude directly
    audio = amplitude * np.sin(2. * np.pi * frequency * t)

    # Test normalization and filtering
    normalized_audio = normalize_audio(audio)
    filtered_audio = high_pass_filter(normalized_audio)
    print("Audio processed successfully.")
    # Verify shapes or some properties
    print(f"Original audio shape: {audio.shape}")
    print(f"Normalized audio shape: {normalized_audio.shape}")
    print(f"Filtered audio shape: {filtered_audio.shape}")
