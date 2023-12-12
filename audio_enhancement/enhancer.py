import librosa
import numpy as np
import soundfile as sf


# Function to load the audio file
def load_audio_file(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)
    return audio, sample_rate

# Function to perform noise reduction (simple version)
def reduce_noise(audio, noise_level=0.01):
    # Assuming a constant noise level, simple noise reduction can be performed
    # by subtracting it from the signal. This is a very rudimentary form of noise reduction.
    return np.clip(audio - noise_level, -1.0, 1.0)

# Function to normalize the volume of the audio clip
def normalize_volume(audio, target_dBFS=-20.0):
    # Compute the current dBFS of the signal
    rms = np.sqrt(np.mean(audio**2))
    current_dBFS = 20 * np.log10(rms)

    # Compute the required gain to achieve the target dBFS
    gain = 10 ** ((target_dBFS - current_dBFS) / 20)

    # Apply the gain to the audio signal
    return audio * gain

# Function to apply a bandpass filter
def bandpass_filter(audio, sample_rate, lowcut=300.0, highcut=3400.0):
    # Apply a butterworth filter
    from scipy.signal import butter, lfilter
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(1, [low, high], btype='band')
    return lfilter(b, a, audio)


file_path = 'audio_sample.mp3'

# Load the audio file
audio, sample_rate = load_audio_file(file_path)

# Perform noise reduction
audio_nr = reduce_noise(audio)

# Normalize volume
audio_normalized = normalize_volume(audio_nr)

# Apply bandpass filter
audio_filtered = bandpass_filter(audio_normalized, sample_rate)

# Now, `audio_filtered` contains the processed audio we will feed to the Whisper model
output_file_path = 'processed_audio.mp3'
sf.write(output_file_path, audio_filtered, sample_rate)