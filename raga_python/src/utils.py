from numpy import arange
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis
import warnings
from pydub import AudioSegment
import io
import soundfile as sf
import sys
warnings.filterwarnings('ignore')

SAMPLE_RATE=44100
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOT_VOICE_TOKEN = '<N>'
END_OF_FILE_TOKEN = '<EOF>'
BLOCK_SIZE = 87 * 30 + 90

NOT_VOICE_TOKEN_MIDI = 2100
MIN_TOKEN = 3600
MAX_TOKEN = 8400
# CLASS NAMES - NOTE: Do not change numbering.
CLASS_NAMES = {
    'saveri': 0,
    'hemavati': 1,
    'thodi': 2,
    'sindhubhairavi': 3
}

def get_current_time_microseconds():
    return int(time.time() * 1e6)

def decompose_note(note: str):
    # Get octave and base note
    if not note:
        return None, None
    octave = int(note[-1]) if note[-1].isdigit() else None
    base = note[:-1] if note[-1].isdigit() else note
    return base, octave

def is_complete_octave(bag: dict):
    # Do we have 7 notes in one octave and one note in the next
    if not bag:
        return False, None
    octaves_present_in_bag = list(sorted(bag.keys()))
    for oct in octaves_present_in_bag[:-1]:
        if len(bag[oct]) >= 7 and len(bag[oct + 1]) >= 1:
            return True, oct
    return False, None

# Calculate the starting timestamp of each frame for which pitch is computed
def get_timestamps(pitches: list, hop_length: int, sr: int) -> list:
    num_frames =  len(pitches)
    return arange(num_frames) * hop_length * 1.0 / sr

def get_tokenizer():
    # Tokenizer
    ALLOWED_TOKENS = []
    for octave in [2,3,4,5]:
        o = str(octave)
        for n in NOTES:
            ALLOWED_TOKENS.append(n+o)

    stoi = {s:i+1 for i,s in enumerate(ALLOWED_TOKENS)}
    stoi[NOT_VOICE_TOKEN] = 0
    itos = {i:s for s,i in stoi.items()}
    vocab_size = len(itos)
    return stoi, itos, vocab_size

def get_tokenizer_midi():
    VALID_TOKENS = set()

    stoi = {}
    itos = {}
    counter = 0
    for v in range(MIN_TOKEN, MAX_TOKEN + 1, 10):
        stoi[v] = counter
        itos[counter] = v
        counter += 1
    return stoi, itos, len(stoi)

def get_classes():
    reverse_map = {}
    for k,v in CLASS_NAMES.items():
        if v in reverse_map:
            raise RuntimeError(f'Duplicate key found {k} -> {v} -> {reverse_map[v]}')
        reverse_map[v] = k
    return reverse_map


def extract_features(y, sr):
    """
    Extract relevant audio features to distinguish between speech and music.
    
    Parameters:
    y: audio time series
    sr: sampling rate (set to 44100 Hz)
    
    Returns:
    dict: Dictionary of extracted features
    """
    features = {}
    
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=2048)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=2048)[0]
    
    # Zero crossing rate (typically higher for speech)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    
    # RMS energy
    rms = librosa.feature.rms(y=y)[0]
    
    # Mel-frequency cepstral coefficients
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048)
    
    # Calculate statistics for each feature
    features['centroid_mean'] = np.mean(spectral_centroids)
    features['centroid_std'] = np.std(spectral_centroids)
    features['rolloff_mean'] = np.mean(spectral_rolloff)
    features['rolloff_std'] = np.std(spectral_rolloff)
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)
    
    # Add MFCC statistics
    for i in range(13):
        features[f'mfcc{i+1}_mean'] = np.mean(mfccs[i])
        features[f'mfcc{i+1}_std'] = np.std(mfccs[i])
    
    return features

def analyze_audio_segments(file_path, segment_duration=10):
    """
    Analyze audio file in segments to detect speech vs music.
    
    Parameters:
    file_path: Path to audio file
    segment_duration: Duration of each segment in seconds
    
    Returns:
    tuple: (classifications, y, sr) where classifications is a list of 0s and 1s
    """
    # Load audio file with specified sampling rate
    y, sr = librosa.load(file_path, sr=44100)
    
    # Calculate number of samples per segment
    samples_per_segment = int(segment_duration * sr)
    
    # Split audio into segments
    segments = []
    for i in range(0, len(y), samples_per_segment):
        segment = y[i:i + samples_per_segment]
        if len(segment) >= sr:  # Only process segments at least 1 second long
            segments.append(segment)
    
    # Analyze each segment
    segment_features = []
    for segment in segments:
        features = extract_features(segment, sr)
        
        # Simple rule-based classification
        # Thresholds adjusted for 44.1kHz sampling rate
        is_music = (
            features['zcr_mean'] < 0.15 and  # Lower ZCR typically indicates music
            features['centroid_std'] > 400 and  # Higher spectral variation for music
            features['rms_std'] > 0.04  # Higher energy variation for music
        )
        
        segment_features.append(1 if is_music else 0)
    
    return segment_features, y, sr

def detect_speech_music(file_path, segment_duration=10, output_path=None):
    """
    Process an audio file, identify speech/music segments, and optionally save music segments.
    
    Parameters:
    file_path: Path to video file
    segment_duration: Duration of each segment in seconds
    output_path: Path where to save the extracted music (if None, music won't be saved)
    
    Returns:
    tuple: (music_percentage, speech_segments, music_segments)
    """
    try:
        # Analyze audio segments
        classifications, y, sr = analyze_audio_segments(file_path, segment_duration)
        
        # Calculate overall statistics
        total_segments = len(classifications)
        music_segments = sum(classifications)
        speech_segments = total_segments - music_segments
        music_percentage = (music_segments / total_segments) * 100
        
        # Get time ranges for each type
        speech_ranges = []
        music_ranges = []
        current_type = classifications[0]
        start_time = 0
        
        for i, c in enumerate(classifications):
            if c != current_type:
                end_time = i * segment_duration
                if current_type == 0:
                    speech_ranges.append((start_time, end_time))
                else:
                    music_ranges.append((start_time, end_time))
                start_time = end_time
                current_type = c
        
        # Add final range
        end_time = len(classifications) * segment_duration
        if current_type == 0:
            speech_ranges.append((start_time, end_time))
        else:
            music_ranges.append((start_time, end_time))
            
        print(f'Classification Complete')
        # If output path is provided, save only the music segments
        if output_path:
            # Convert to AudioSegment for MP3 handling
            # First, save the numpy array as WAV in memory
            wav_io = io.BytesIO()
            sf.write(wav_io, y, sr, format='WAV')
            wav_io.seek(0)
            audio = AudioSegment.from_wav(wav_io)
            
            # Extract and concatenate music segments
            music_segments = AudioSegment.empty()
            for start, end in music_ranges:
                start_ms = int(start * 1000)  # Convert to milliseconds
                end_ms = int(end * 1000)
                music_segments += audio[start_ms:end_ms]
            
            # Export as MP3
            music_segments.export(output_path, format="mp3")
            print(f"Music segments saved to {output_path}")
        
        return music_percentage, speech_ranges, music_ranges
    
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(f"Exception occurred on line {exc_tb.tb_lineno}")
        print(f'Exception type: {exc_type}')
        print(f"Error processing file: {str(e)}")
        return None