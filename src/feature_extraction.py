import librosa
import numpy as np

def extract_features(audio, sr=44100, n_mfcc=13, frame_size=2048, hop_size=512):
    """Extracts features from an audio signal.

    Args:
        audio (np.ndarray): The audio data.
        sr (int, optional): The sample rate. Defaults to 44100.
        n_mfcc (int, optional): The number of MFCCs to extract. Defaults to 13.
        frame_size (int, optional): The frame size for feature extraction. Defaults to 2048.
        hop_size (int, optional): The hop size for feature extraction. Defaults to 512.

    Returns:
        dict: A dictionary of extracted features.
    """
    features = {}

    # Mel-Frequency Cepstral Coefficients (MFCCs)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=frame_size, hop_length=hop_size)
    features['mfcc'] = np.mean(mfccs.T, axis=0)

    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=frame_size, hop_length=hop_size)
    features['spectral_centroid'] = np.mean(spectral_centroid)

    # Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=frame_size, hop_length=hop_size)
    features['spectral_bandwidth'] = np.mean(spectral_bandwidth)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=frame_size, hop_length=hop_size)
    features['zcr'] = np.mean(zcr)

    return features

if __name__ == '__main__':
    # Example usage
    # Create a dummy audio file for testing
    import soundfile as sf
    sr = 44100
    duration = 5
    frequency = 440
    t = np.linspace(0., duration, int(sr * duration))
    amplitude = np.iinfo(np.int16).max * 0.5
    data = amplitude * np.sin(2. * np.pi * frequency * t)

    # In-memory audio data
    audio_data = data.astype(np.float32) / np.iinfo(np.int16).max


    features = extract_features(audio_data, sr=sr)
    print("Features extracted successfully:")
    for feature_name, feature_value in features.items():
        print(f"- {feature_name}: {feature_value}")
