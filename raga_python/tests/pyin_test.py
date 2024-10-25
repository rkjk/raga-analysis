import unittest

from src.pyin_pitch_detect import *

import librosa

class PYINTest(unittest.TestCase):
    def test_10_seconds_saveri_alapana_tnseshagopalan(self):
        RATE = 44100
        FRAME_LENGTH = 2048
        HOP_LENGTH = 512
        pyin_pitch_detector = PYINPitchDetect(RATE, frame_length=2048, hop_length=512)
        audio, sr = librosa.load('../data/TN Seshagopalan - Saveri Alapana [QDw-jpTw3Q4].wav', sr=44100, duration=10.0)
        pyin = PYINPitchDetect(sr, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        pitches, voiced_flag, voiced_prob = pyin.detect(audio)

        music_pitches = []
        music_probs = []
        for i in range(len(pitches)):
            if voiced_flag[i]:
                music_pitches.append(librosa.hz_to_note(pitches[i]))
                music_probs.append(voiced_prob[i])
        last_seen = None
        compressed_runs = []
        for pitch in music_pitches:
            if last_seen is None or pitch != last_seen:
                compressed_runs.append(pitch)
                last_seen = pitch
        with open("tests/test_10_seconds_saveri_alapana_tnseshagopalan.txt", "w") as f:
            f.writelines([item + "\n" for item in compressed_runs])