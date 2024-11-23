import pyaudio
import numpy as np
import librosa
from queue import Queue
import threading
import time
from dataclasses import dataclass
from librosa import pyin, note_to_hz
import os

import torch

from pyin_pitch_detect import *
import utils

from model.cnn1d import ConvNet_1D

audio_queue = Queue()

import torch.nn as nn
import torch.nn.functional as F

# Tokenizer
stoi, itos, vocab_size = utils.get_tokenizer()
raga_map = utils.get_classes()

def process_audio():
    chunk_size = utils.SAMPLE_RATE  # 30 seconds of audio
    buffer = []

    ### Pitch Detector
    pitch_detector = PYINPitchDetect(utils.SAMPLE_RATE, frame_length=2048, hop_length=512)


    # Model
    in_channels = 1  # Input channels (e.g., single-channel audio)
    out_channels = 2  # Output channels (e.g., regression output)
    kernel_size = 3   # Kernel size
    stride = 1       # Stride
    padding = 0      # Padding
    n_tokens = vocab_size
    n_embd = 24

    # Learning rate
    lr = 0.001
    epochs = 0
    model = ConvNet_1D(in_channels, out_channels, kernel_size, n_embd, n_tokens, device='cpu')
    MODEL_PATH = './models/simple-test-model'
    checkpoint = torch.load(MODEL_PATH)
    epochs = checkpoint['epochs']
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']
    print(f'checkpoint after epoch: {epochs}')
    print(f'train loss: {train_loss}')
    print(f'val loss: {val_loss}')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    music_pitches = []
    while True:
        while len(buffer) < chunk_size:
            try:
                buffer.extend(audio_queue.get(timeout=1))
            except:
                # Handle the case where there isn't enough data yet
                print(f'Not enough data')
                time.sleep(0.1)
                continue
        # Process the 30-second chunk
        pitches, voiced_flag, voiced_prob = pitch_detector.detect(np.array(buffer))
        buffer = []
        for i in range(len(pitches)):
            svara = utils.NOT_VOICE_TOKEN
            if voiced_flag[i] and voiced_prob[i] > 0.5:
                svara = librosa.hz_to_note(pitches[i]).replace('♯', '#')
            music_pitches.append(svara)
        prev_data = None
        print(f'len of data: {len(music_pitches)}')
        if len(music_pitches) >= 2700:
            data = music_pitches[:2700]
            music_pitches = music_pitches[2700:]
            prev_data = data
            data = [stoi[t] for t in data]
            not_voice_count = data.count(0)
            if not_voice_count >= 2600:
                print(f'Prediction: NOT_VOICE. Count {not_voice_count}')
                continue
            print(data)
            with torch.no_grad():
                data = torch.tensor(data).view(1,-1)
                logits = model(data)
                probabilities = torch.softmax(logits, dim=1)
                print(f'probs: {probabilities}')
                max_prob, max_index = logits.max(dim=1)
                print(f'Prediction: {raga_map[max_index.item()]}, prediction_prob: {max_prob}')
        #print(f'Processed chunk. Pitches len {pitches.shape}, {len(voiced_flag)}, {len(voiced_prob)}')

def capture_audio():
    RATE = 44100
    #CHUNK = int (0.010 * RATE) * 4  # 20 ms chunks x 4 bytes per sample
    CHUNK=RATE
    # Read 1 second of data in one frame
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
    while True:
        data = np.frombuffer(stream.read(CHUNK), dtype=np.float32)
        audio_queue.put(data)
    stream.stop_stream()
    stream.close()
    p.terminate()

# Start the audio capture in a separate thread
capture_thread = threading.Thread(target=capture_audio)
capture_thread.daemon = True  # So that the thread dies when the main thread dies
capture_thread.start()
print(f'Started audio thread')

# Start the processing in a separate thread
process_thread = threading.Thread(target=process_audio)
process_thread.daemon = True  # So that the thread dies when the main thread dies
process_thread.start()
print(f'Started process thread')

# Keep the main thread alive
while True:
    time.sleep(1)