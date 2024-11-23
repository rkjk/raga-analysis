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

audio_queue = Queue()

import torch.nn as nn
import torch.nn.functional as F

# Tokenizer
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOT_VOICE_TOKEN = '<N>'
END_OF_FILE_TOKEN = '<EOF>'

ALLOWED_TOKENS = []
for octave in [2,3,4,5]:
    o = str(octave)
    for n in NOTES:
        ALLOWED_TOKENS.append(n+o)

stoi = {s:i+1 for i,s in enumerate(ALLOWED_TOKENS)}
stoi[NOT_VOICE_TOKEN] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)

class ConvNet_1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_embd, n_tokens, device='cpu', dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(n_tokens, n_embd, device=device)
        self.ConvNet = nn.Sequential(
            nn.Conv1d(n_embd, 32, kernel_size=kernel_size, device=device),
            nn.ReLU(),
            nn.BatchNorm1d(32, device=device),
            nn.MaxPool1d(kernel_size=5),

            nn.Conv1d(32, 64, kernel_size=kernel_size, device=device),
            nn.ReLU(),
            nn.BatchNorm1d(64, device=device),
            nn.MaxPool1d(kernel_size=5),

            nn.Conv1d(64, 128, kernel_size=kernel_size, device=device),
            nn.ReLU(),
            nn.BatchNorm1d(128, device=device),
            nn.MaxPool1d(kernel_size=5),

            nn.Conv1d(128, 256, kernel_size=kernel_size, device=device),
            nn.ReLU(),
            nn.BatchNorm1d(256, device=device),
            nn.MaxPool1d(kernel_size=5),
            nn.Dropout(dropout),

            nn.Flatten()
        )

         #Fully connected layers for regression or classification
        self.task = nn.Sequential(
            nn.Linear(768, 100, device=device),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(100, out_channels, device=device)
        )

    def calculate_output_size(self, in_channels, kernel_size, stride, padding):
        # Calculate the output size after multiple convolutional and pooling layers
        output_size = (in_channels - kernel_size + 2 * padding) // stride + 1
        #print(f'output_size {output_size}')
        output_size = (output_size - kernel_size + 2 * padding) // stride + 1
        #print(f'output_size {output_size}')
        output_size = (output_size - kernel_size + 2 * padding) // stride + 1
        #print(f'output_size {output_size}')
        output_size = (output_size - kernel_size + 2 * padding) // stride + 1
        #print(f'output_size {output_size}')
        return output_size

    def forward(self, x):
        x = self.emb(x)
        sequence_length = x.shape[2]
        minibatch_length = x.shape[0]
        x = x.view(minibatch_length, sequence_length, -1)
        x = self.ConvNet(x)
        output = self.task(x)
        return output

def process_audio():
    SAMPLE_RATE = 44100
    chunk_size = SAMPLE_RATE  # 30 seconds of audio
    buffer = []

    ### Pitch Detector
    pitch_detector = PYINPitchDetect(SAMPLE_RATE, frame_length=2048, hop_length=512)


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
        #print(f'buffer shape {buffer.shape}')
        #chunk = np.array(buffer[:chunk_size])
        # Your processing logic here
        pitches, voiced_flag, voiced_prob = pitch_detector.detect(np.array(buffer))
        buffer = []
        for i in range(len(pitches)):
            svara = NOT_VOICE_TOKEN
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
                print(f'Prediction: {max_index}, prediction_prob: {max_prob}')
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