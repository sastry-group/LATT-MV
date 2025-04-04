import numpy as np
import librosa
import scipy.signal

from moviepy.editor import VideoFileClip

def create_audio_file(path, audio_path):
    # Load the video file
    video = VideoFileClip(path)

    # Extract the audio
    audio = video.audio

    # Save the audio to a file
    audio.write_audiofile(audio_path)

def get_times(audio_path):

    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    # Compute the envelope of the signal
    hop_length = 256
    frame_length = 1024
    envelope = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length))
    envelope = np.sum(envelope, axis=0)

    # Apply a high-pass filter to remove low-frequency noise
    b, a = scipy.signal.butter(5, 1000 / (sr / 2), btype='high')
    filtered_envelope = scipy.signal.filtfilt(b, a, envelope)

    # Detect peaks in the filtered envelope
    distance = int(sr * 0.1 / hop_length)  # Adjust distance based on expected minimum time between bounces
    peaks, _ = scipy.signal.find_peaks(filtered_envelope, height=np.mean(filtered_envelope), distance=distance)

    # Convert peak indices to time
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)

    return times