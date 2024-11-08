from src.pyin_pitch_detect import *
from src.chroma_detect import *
from src.utils import *

import os
import librosa
from numpy import argmax
import noisereduce as nr
import soundfile as sf
import pytest

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
    @pytest.mark.skip(reason="Temporarily disabled for demonstration purposes")
    def test_10_seconds_saveri_alapana_tnseshagopalan(self):
        RATE = 44100
        FRAME_LENGTH = 2048
        HOP_LENGTH = 512
        audio, sr = librosa.load('../data/TN Seshagopalan - Saveri Alapana [QDw-jpTw3Q4].wav', sr=44100, duration=10.0)
        pyin = PYINPitchDetect(sr, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        pitches, voiced_flag, voiced_prob = pyin.detect(audio)

        timestamps = get_timestamps(pitches, HOP_LENGTH, RATE)

        music_pitches = []
        music_probs = []
        for i in range(len(pitches)):
            timestamp = timestamps[i]
            vp = voiced_prob[i]
            svara = ''
            if voiced_flag[i]:
                note = librosa.hz_to_note(pitches[i]).replace('♯', '#')
                svara = note_to_svara[note]
            if vp > 0.5:
                music_pitches.append((timestamp, vp, svara))
        with open("tests/test_10_seconds_saveri_alapana_tnseshagopalan.txt", "w") as f:
            f.writelines([str(round(item[0], 3)) + ',' + str(round(item[1], 3)) + "," + item[2] + "\n" for item in music_pitches])
    
    def test_mayamalavagowlai_ragasurabhi(self):
        RATE = 44100
        FRAME_LENGTH = 8192
        HOP_LENGTH = 4096
        audio, sr = librosa.load('../data/raga-mayamalavagowlai-arohanam_avarohanam.mp3', sr=44100)
        pyin = PYINPitchDetect(sr, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        pitches, voiced_flag, voiced_prob = pyin.detect(audio)

        timestamps = get_timestamps(pitches, HOP_LENGTH, RATE)
        #print(f"mg - tlen {len(timestamps)}")
        #print(f"mg - voiced {len(voiced_flag)}")
        music_pitches = []
        music_probs = []
        for i in range(len(pitches)):
            timestamp = timestamps[i]
            vp = voiced_prob[i]
            svara = ''
            if voiced_flag[i]:
                #note = librosa.hz_to_note(pitches[i]).replace('♯', '#')
                #svara = note_to_svara[note]
                svara = str(round(pitches[i], 2))
            if vp > 0.5:
                music_pitches.append((timestamp, vp, svara))
        with open("tests/test_mayamalavagowlai_ragasurabhi.txt", "w") as f:
            f.writelines([str(round(item[0], 3)) + ',' + str(round(item[1], 3)) + "," + item[2] + "\n" for item in music_pitches])

    def test_mayamalavagowlai_ragasurabhi_chroma(self):
        RATE = 44100
        FRAME_LENGTH = 8192
        HOP_LENGTH = 4096
        pitch_detector = ChromaDetector(RATE, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        audio, sr = librosa.load('../data/raga-mayamalavagowlai-arohanam_avarohanam.mp3', sr=44100)
        chroma = ChromaDetector(sr, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        pitches = chroma.detect(audio)

        pitches = argmax(pitches, axis=0)
        timestamps = get_timestamps(pitches, HOP_LENGTH, RATE)
        music_pitches = []
        for i in range(len(pitches)):
            music_pitches.append((timestamps[i], pitches[i]))

        with open("tests/test_mayamalavagowlai_ragasurabhi_chroma.txt", "w") as f:
            f.writelines([str(round(item[0], 3)) + ',' + str(round(item[1], 3)) + "\n" for item in music_pitches])

    @pytest.mark.skip(reason="Temporarily disabled for demonstration purposes")
    def test_mridangam_only(self):
        RATE = 44100
        FRAME_LENGTH = 8192
        HOP_LENGTH = 4096
        audio, sr = librosa.load('../data/sai-giridhar-mridangam.wav', sr=44100)
        pyin = PYINPitchDetect(sr, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        pitches, voiced_flag, voiced_prob = pyin.detect(audio)

        timestamps = get_timestamps(pitches, HOP_LENGTH, RATE)

        music_pitches = []
        music_probs = []
        for i in range(len(pitches)):
            timestamp = timestamps[i]
            vp = voiced_prob[i]
            svara = ''
            if voiced_flag[i]:
                note = librosa.hz_to_note(pitches[i]).replace('♯', '#')
                svara = note_to_svara[note]
            if vp > 0.5:
                music_pitches.append((timestamp, vp, svara))
        with open("tests/sai-giridhar-mridangam.txt", "w") as f:
            f.writelines([str(round(item[0], 3)) + ',' + str(round(item[1], 3)) + "," + item[2] + "\n" for item in music_pitches])

    def test_mayamalavagowlai_raghav_scales(self):
        RATE = 44100
        FRAME_LENGTH = 8192
        HOP_LENGTH = 4096

        directory = '../data/raghav-mayamalavagowlai-scales'

        files = {}

        all_files = os.listdir(directory)
        print(all_files)
        for filename in all_files:
            if filename.startswith('nr_'):
                continue
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                print(filepath)
                # Open and process the file if needed
                audio, sr = librosa.load(filepath, sr=RATE)
                #audio = nr.reduce_noise(y=audio,sr=sr)
                fname = '.'.join(filename.split(".")[:-1])
                #output_fname = 'nr_' + fname + '.wav'
                #if output_fname not in all_files:
                #   sf.write(os.path.join(directory, output_fname), audio, sr)
                #else:
                #    print(f'{output_fname} is already present')
                files[fname] = audio


        chroma = ChromaDetector(RATE, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)

        pitches = {}
        for k, v in files.items():
            p = argmax(chroma.detect(v), axis=0)
            t = get_timestamps(p, HOP_LENGTH, RATE)
            pitches[k] = (p, t)
        
        for k, v in pitches.items():
            mp = list(zip(v[1], v[0]))
            fname = "test_raghav_scales_" + k
            with open("tests/" + fname, "w") as f:
                f.writelines([str(round(item[0], 3)) + ',' + str(round(item[1], 3)) + "\n" for item in mp])
