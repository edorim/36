# Fart-to-MIDI Audio Pipeline

This project is a Python-based audio processing pipeline that automatically transcribes flatulence events from an audio recording into musical information in the MIDI format.

## Overview of the Pipeline

The Fart-to-MIDI pipeline is an innovative system designed to identify, label, and translate fart sounds into musical notes. It leverages a standard machine learning approach for audio classification and segmentation, followed by a unique mapping layer for musical conversion.

The pipeline consists of six main stages:

1.  **Audio Acquisition & Normalization**: Captures and prepares high-quality audio for analysis.
2.  **Preprocessing & Feature Extraction**: Converts raw audio data into a format suitable for machine learning.
3.  **Fart Identification and Labeling (Classification)**: Identifies and categorizes flatulence events.
4.  **Transient Analysis and Segmentation**: Extracts musical parameters from the identified events.
5.  **MIDI Mapping and Transcription**: Converts acoustic features into standardized MIDI messages.
6.  **MIDI Output Generation**: Creates a MIDI file from the transcribed musical information.

## Getting Started

### Prerequisites

- Python 3.7+
- The Python libraries listed in `requirements.txt`

### Installation

1.  Clone this repository:
    ```bash
    git clone https://github.com/your-username/fart-to-midi.git
    ```
2.  Navigate to the project directory:
    ```bash
    cd fart-to-midi
    ```
3.  Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

You can run the pipeline using the `main.py` script. It takes two arguments: the path to the input audio file and the path for the output MIDI file.

```bash
python main.py <input_audio_file> <output_midi_file>
```

For example:

```bash
python main.py data/my_audio.wav output/transcribed_farts.mid
```

If the specified input audio file does not exist, the script will create a dummy audio file for testing purposes.

## Project Structure

- `main.py`: The main script to run the entire pipeline.
- `src/`: Contains the core Python modules for each stage of the pipeline.
  - `normalization.py`: Handles audio loading, normalization, and filtering.
  - `feature_extraction.py`: Extracts features like MFCCs from the audio.
  - `classification.py`: A placeholder for the machine learning classification model.
  - `analysis.py`: Analyzes audio segments to extract pitch, duration, and volume.
  - `midi_mapping.py`: Maps the extracted features to MIDI notes and saves the MIDI file.
- `data/`: A directory for storing input audio files.
- `output/`: The directory where the generated MIDI files are saved.
- `requirements.txt`: A list of the required Python libraries.

## Future Development

This project is a proof-of-concept and can be extended in several ways:

- **Train a Machine Learning Model**: The current classification stage uses a placeholder. A real model could be trained on a labeled dataset of flatulence sounds.
- **Real-time Processing**: The pipeline could be adapted to process audio in real-time from a microphone.
- **Advanced MIDI Mapping**: The MIDI mapping could be made more sophisticated, with different instruments or musical scales for different types of farts.
