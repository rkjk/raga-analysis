import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import librosa
from collections import deque
import time

def get_current_time_microseconds():
    return int(time.time() * 1e6)

# Set up audio parameters
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
CHUNK = int (0.020 * RATE) * 4  # 20 ms chunks x 4 bytes per sample

# Set up plotting parameters
fig, ax = plt.subplots()
ax.set_title('Pitch Over Time')
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Pitch (Hz)')
plt.ion()  # Turn on interactive mode

# Initialize PyAudio
p = pyaudio.PyAudio()

history = deque(maxlen=1024)  # Store past pitches for smoothing
start_time = None

# Open the audio stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Starting real-time pitch detection...")

while True:
    # Read audio data from the stream
    data = np.frombuffer(stream.read(CHUNK), dtype=np.float32)
    #print(f'data size = {data.shape}')
    
    # Detect pitch using the PYin algorithm
    ff, voiced_flag, voiced_prob = librosa.pyin(
        data,
        fmin = librosa.note_to_hz('A2'), 
        fmax=librosa.note_to_hz('G#5'), 
        sr=RATE,
        frame_length=CHUNK)
    #print(f'ff size = {len(ff)}')

    # Print the detected pitch
    for k in range(len(ff)):
        if voiced_flag[k] == True and voiced_prob[k] > 0.5:
            print(f"Detected note: {librosa.hz_to_note(ff[k])}")

# Close the audio stream and PyAudio
stream.stop_stream()
stream.close()
p.terminate()

