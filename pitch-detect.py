import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import librosa
from collections import defaultdict
import time
from dataclasses import dataclass

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

# Set up audio parameters
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
CHUNK = int (0.020 * RATE) * 4  # 20 ms chunks x 4 bytes per sample

class PYINPitchDetect:
    def __init__(self, sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), frame_length=None, hop_length=None):
        if not sr:
            raise RuntimeError("Sample rate not specified")
        if not frame_length:
            frame_length = 1024
        self.sr = sr
        self.fmin = fmin
        self.fmax = fmax
        self.frame_length = frame_length
        self.hop_length = hop_length
    
    def detect(self, data):
        return librosa.pyin(
                data,
                fmin=self.fmin, 
                fmax=self.fmax,
                sr=self.sr,
                frame_length=self.frame_length,
                hop_length=self.hop_length)
    
    def name(self):
        return "PYIN"

class LiveAudioDetect:
    def __init__(self, forma, channels, rate, frames_per_buffer, pitch_detector):
        self.format = forma
        self.channels = channels
        self.rate = rate
        self.frames_per_buffer = frames_per_buffer
        self.initialize()
        self.bag_of_notes = defaultdict(set)
        self.pitch_detector = pitch_detector
        
    def __del__(self):
        # Close the audio stream
        if self.p is not None:
            self.p.terminate()

    def initialize(self):
        try:
            self.p = pyaudio.PyAudio()
        except Exception as e:
            err = f"Could not initialize PyAudio: {e}"
            raise RuntimeError(err)
        
        # Open the audio stream

    def start(self):
        stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.frames_per_buffer)
        print("Starting real-time pitch detection...")
        while True:
        # Read audio data from the stream - We need roughly frames per second
        # we get rate samples per second
            data = np.frombuffer(stream.read(self.rate), dtype=np.float32)
        
            # Detect pitch using the PYin algorithm
            ff, voiced_flag, voiced_prob = self.pitch_detector.detect(data)

            # Print the detected pitch
            for k in range(len(ff)):
                if voiced_flag[k] == True and voiced_prob[k] > 0.5:
                    base, octave = decompose_note(librosa.hz_to_note(ff[k]))
                    if octave is not None:
                        self.bag_of_notes[octave].add(base)
                        flag, oct = is_complete_octave(self.bag_of_notes)
                        #if flag:
                            #print(f"Svaras sung: {list(sorted(bag_of_notes[oct]))}")
                    print(f"Detected note: {librosa.hz_to_note(ff[k])}")

        stream.stop_stream()
        stream.close()
        self.p.terminate()

if __name__ == "__main__":
    # Set up audio parameters
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 44100
    FRAME_LENGTH = 2048
    HOP_LENGTH = 512
    CHUNK = int (0.020 * RATE) * 4  # 20 ms chunks x 4 bytes per sample
    pyin_pitch_detector = PYINPitchDetect(RATE, frame_length=2048, hop_length=512)
    detect = LiveAudioDetect(FORMAT, CHANNELS, RATE, CHUNK, pyin_pitch_detector)
    detect.start()