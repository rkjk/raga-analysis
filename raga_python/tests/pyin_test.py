from src.pyin_pitch_detect import *

import librosa

seshu_map = {
    'C#3': 'M1', 'D3': 'R1', 'D#3': 'R2', 'E3': 'G1', 'F3': 'G2', 'F#3': 'G3',
    'G3': 'M2', 'G#3': 'P', 'A3': 'D1', 'A#3': 'D2', 'B3': 'N1',
    'C#4': 'N2', 'D4': 'N3', 'D#4': 'S', 'E4': 'R1', 'F4': 'R2', 'F#4': 'G1',
    'G4': 'G2', 'G#4': 'G3', 'A4': 'M2', 'A#4': 'P', 'B4': 'D1',
    'C#5': 'D2', 'D5': 'N1', 'D#5': 'N2', 'E5': 'N3', 'F5': 'S', 'F#5': 'R1',
    'G5': 'R2', 'G#5': 'G1', 'A5': 'G2', 'A#5': 'G3', 'B5': 'M2',
    'C#6': 'P', 'D6': 'D1', 'D#6': 'D2', 'E6': 'N1', 'F6': 'N2', 'F#6': 'N3',
    'G6': 'S'
}

# Map Western notes to Carnatic svaras
note_to_svara = {
    'C2': 'M1', 'C#2': 'R1', 'D2': 'R2', 'D#2': 'R3', 'E2': 'G1', 'F2': 'G2', 'F#2': 'G3',
    'G2': 'M2', 'G#2': 'P', 'A2': 'D1', 'A#2': 'D2', 'B2': 'D3',
    'C3': 'N1', 'C#3': 'N2', 'D3': 'N3', 'D#3': 'S', 'E3': 'R1', 'F3': 'R2', 'F#3': 'R3',
    'G3': 'G1', 'G#3': 'G2', 'A3': 'G3', 'A#3': 'M2', 'B3': 'P',
    'C4': 'D1', 'C#4': 'D2', 'D4': 'D3', 'D#4': 'N1', 'E4': 'N2', 'F4': 'N3', 'F#4': 'S',
    'G4': 'R1', 'G#4': 'R2', 'A4': 'R3', 'A#4': 'G1', 'B4': 'G2',
    'C5': 'G3', 'C#5': 'M2', 'D5': 'P', 'D#5': 'D1', 'E5': 'D2', 'F5': 'D3', 'F#5': 'N1',
    'G5': 'N2', 'G#5': 'N3', 'A5': 'S', 'A#5': 'R1', 'B5': 'R2',
    'C6': 'R3', 'C#6': 'G1', 'D6': 'G2', 'D#6': 'G3', 'E6': 'M2', 'F6': 'P', 'F#6': 'D1',
    'G6': 'D2', 'G#6': 'D3', 'A6': 'N1', 'A#6': 'N2', 'B6': 'N3',
    'C7': 'S'
}

class TestPYIN:
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
                note = librosa.hz_to_note(pitches[i]).replace('♯', '#')
                svara = note_to_svara[note]
                music_pitches.append(svara)
                music_probs.append(voiced_prob[i])
        last_seen = None
        compressed_runs = []
        for svara in music_pitches:
            if last_seen is None or svara != last_seen:
                compressed_runs.append(svara)
                last_seen = svara
        with open("tests/test_10_seconds_saveri_alapana_tnseshagopalan.txt", "w") as f:
            f.writelines([item + "\n" for item in compressed_runs])
    
    def test_mayamalavagowlai_ragasurabhi(self):
        RATE = 44100
        FRAME_LENGTH = 2048
        HOP_LENGTH = 512
        pyin_pitch_detector = PYINPitchDetect(RATE, frame_length=2048, hop_length=512)
        audio, sr = librosa.load('../data/raga-mayamalavagowlai-arohanam_avarohanam.mp3', sr=44100)
        pyin = PYINPitchDetect(sr, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        pitches, voiced_flag, voiced_prob = pyin.detect(audio)

        music_pitches = []
        music_probs = []
        for i in range(len(pitches)):
            if voiced_flag[i]:
                note = librosa.hz_to_note(pitches[i]).replace('♯', '#')
                svara = note_to_svara[note]
                music_pitches.append(svara)
                music_probs.append(voiced_prob[i])
        last_seen = None
        compressed_runs = []
        for svara in music_pitches:
            if last_seen is None or svara != last_seen:
                compressed_runs.append(svara)
                last_seen = svara
        with open("tests/test_mayamalavagowlai_ragasurabhi.txt", "w") as f:
            f.writelines([item + "\n" for item in compressed_runs])