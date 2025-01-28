import librosa
import numpy as np
import io
import soundfile as sf
from pydub import AudioSegment
import sys
import csv
import tensorflow as tf
import tensorflow_hub as hub

# Load YAMNet model once
YAMNET_MODEL = hub.load('https://tfhub.dev/google/yamnet/1')

def classify_segment_yamnet(audio, sr, carnatic_threshold=0.2):
    """Classify audio segment for Carnatic music probability"""
    # Resample to 16kHz if needed
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    # Audio preprocessing
    audio = audio.astype(np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    
    # Run YAMNet inference
    scores, _, _ = YAMNET_MODEL(audio)
    
    # Get relevant class indices
    carnatic_idx, music_idx, supporting_indices = get_indices()
    
    # Calculate combined probability
    carnatic_score = np.mean(scores[:, carnatic_idx])
    scores_np = scores.numpy()
    supporting_score = np.mean(np.sum(scores_np[:, supporting_indices], axis=1))
    
    # Weighted combination (adjust weights based on your observations)
    total_score = 0.5 * carnatic_score + 0.5 * supporting_score
    
    # Additional music context check
    music_score = np.mean(scores[:, music_idx])  # General music class
    
    # Final classification
    is_carnatic = (total_score > carnatic_threshold) and (music_score > 0.2)
    
    return 1 if is_carnatic else 0

def get_indices():
    class_map_path = YAMNET_MODEL.class_map_path().numpy()
    class_names = class_names_from_csv(class_map_path)
    music_idx = class_names.index('Music')
    carnatic_idx = class_names.index('Carnatic music')

    support = ['Singing', 'Musical instrument', 'Violin, fiddle', 'Bowed string instrument', 'Plucked string instrument', 'Chant', 'Vocal music', 'Music of Asia', 'Traditional music']
    support_idxs = []
    for s in support:
        support_idxs.append(class_names.index(s))
    
    return carnatic_idx, music_idx, support_idxs

def analyze_audio_segments(file_path, segment_duration=10):
    """
    Analyze audio file in segments using YAMNet
    """
    # Load audio at 16kHz for YAMNet compatibility
    y, sr = librosa.load(file_path, sr=16000)
    
    # Calculate samples per segment
    samples_per_segment = int(segment_duration * sr)
    
    # Split audio into segments
    segments = []
    for i in range(0, len(y), samples_per_segment):
        segment = y[i:i + samples_per_segment]
        if len(segment) >= sr:  # At least 1 second
            segments.append(segment)
    
    # Classify segments
    classifications = []
    for segment in segments:
        classifications.append(classify_segment_yamnet(segment, sr))
    
    return classifications, y, sr

def detect_speech_music(file_path, segment_duration=10, output_path=None):
    """
    Process audio file using YAMNet, maintain original interface
    """
    try:
        classifications, y, sr = analyze_audio_segments(file_path, segment_duration)
        
        # Calculate statistics
        total_segments = len(classifications)
        music_segments = sum(classifications)
        speech_segments = total_segments - music_segments
        music_percentage = (music_segments / total_segments) * 100
        
        # Generate time ranges
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
            
        print('Classification Complete')
        
        # Save music segments if requested
        if output_path:
            wav_io = io.BytesIO()
            sf.write(wav_io, y, sr, format='WAV')
            wav_io.seek(0)
            audio = AudioSegment.from_wav(wav_io)
            
            music_audio = AudioSegment.empty()
            for start, end in music_ranges:
                start_ms = int(start * 1000)
                end_ms = int(end * 1000)
                music_audio += audio[start_ms:end_ms]
            
            music_audio.export(output_path, format="mp3")
            print(f"Music segments saved to {output_path}")
        
        return music_percentage, speech_ranges, music_ranges
    
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(f"Exception occurred on line {exc_tb.tb_lineno}")
        print(f'Exception type: {exc_type}')
        print(f"Error processing file: {str(e)}")
        return None

def class_names_from_csv(class_map_csv_text):
  """Returns list of class names corresponding to score vector."""
  class_names = []
  with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      class_names.append(row['display_name'])

  return class_names