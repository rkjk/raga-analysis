from src.pyin_pitch_detect import *
from src.chroma_detect import *
from src.utils import *

import os
import time
import librosa
from numpy import argmax
import noisereduce as nr
import soundfile as sf
import concurrent.futures
import math

import matplotlib.pyplot as plt

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

NOT_VOICE_TOKEN = '<N>'

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
                #svara = note_to_svara[note]
                svara = note
            if vp > 0.5:
                music_pitches.append((timestamp, vp, svara))
            else:
                music_pitches.append((timestamp, vp, NOT_VOICE_TOKEN))
        with open("tests/test_10_seconds_saveri_alapana_tnseshagopalan.txt", "w") as f:
            f.writelines([str(round(item[0], 3)) + ',' + str(round(item[1], 3)) + "," + item[2] + "\n" for item in music_pitches])

    def pitch_shift_helper(self, audio, sr, n_steps, outfile):
        shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
        print(f'Shifted {outfile}')
        pitches, voiced_flag, voiced_prob = librosa.pyin(
                shifted,
                fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C6'),
                sr=44100,
                frame_length=2048,
                hop_length=512
            )
        print(f'generated pitches for {outfile}')
        #sf.write(outfile, shifted, sr)

    @pytest.mark.skip(reason="Temporarily disabled for demonstration purposes")
    def test_stereo(self):
        RATE = 44100
        FRAME_LENGTH = 2048
        HOP_LENGTH = 512
        audio1, sr = librosa.load('../data/simple-test/input/saveri/nedunuri-amma-nannubrovave.mp3', sr=44100)
        audio2, sr = librosa.load('../data/simple-test/input/saveri/nedunuri-amma-nannubrovave.mp3', sr=44100, mono=False)
        pyin = PYINPitchDetect(sr, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)

        #timestamps = get_timestamps(pitches1, HOP_LENGTH, RATE)
        pitches1, voiced_flag1, voiced_prob1 = pyin.detect(audio1)
        pitches2, voiced_flag2, voiced_prob2 = pyin.detect(audio2)
        print(f'Pitches1: {pitches1.shape}')
        print(f'Pitches2: {pitches2.shape}')
        diff = 0
        same = 0
        not_voice = 0
        for i in range(len(pitches1)):
            #print(i)
            if voiced_flag1[i] and voiced_prob1[i] > 0.5:
                note1 = librosa.hz_to_note(pitches2[0][i]).replace('♯', '#')
                note2 = librosa.hz_to_note(pitches2[1][i]).replace('♯', '#')
                #print(f'note1: {note1}, {note2}')
                if note1 != note2:
                    diff += 1
                else:
                    same += 1
            else:
                not_voice += 1
        print(f'On stereo: same {same}, diff {diff} not voice {not_voice}')
        #with open("tests/test_mono.txt", "w") as f, open("tests/test_stereo.txt", "w") as g:
        #    for i in range(len(pitches1)):
                

    #@pytest.mark.skip(reason="Temporarily disabled for demonstration purposes")
    def test_midi(self):
        RATE = 44100
        FRAME_LENGTH = 2048
        HOP_LENGTH = 512

        duration = 10.0
        audio, sr = librosa.load('../data/TN Seshagopalan - Saveri Alapana [QDw-jpTw3Q4].wav', sr=44100, duration=duration)
        #audio, sr = librosa.load('../data/simple-test/thodi-vittal-rangan.mp3', sr=44100, duration=duration)

        pyin = PYINPitchDetect(sr, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)

        pitches, voiced_flag, voiced_prob = pyin.detect(audio)
        timestamps = get_timestamps(pitches, HOP_LENGTH, RATE)

        sound_pitches = []
        unvoice_prob = []

        NAN = librosa.midi_to_hz(21)
        VOICE_PROB_THRESHOLD = 0.3
        for i in range(len(pitches)):
            if voiced_prob[i] > VOICE_PROB_THRESHOLD and not math.isnan(pitches[i]):
                sound_pitches.append(pitches[i])
            else:
                sound_pitches.append(NAN) # Corresponds to MIDI number 21
        print(f'Pitches: {sound_pitches}')
        midi = librosa.hz_to_midi(sound_pitches)
        print(f'After midi conversion: {midi}')
        midi_cents = np.round([m * 100 for m in midi])
        print(f'midi_cents: {midi_cents}')
        return
        mask = midi_cents < 3700

        total_masked = sum(mask)
        print(f'Number of masked={sum(mask)}')
        #print(f'pitches shape: {sound_pitches.shape}, midi shape: {midi.shape}')
        #print(f'pitches: {sound_pitches}')

        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        xaxis = np.linspace(0, duration, len(midi_cents))
        ax.plot(xaxis[mask], midi_cents[mask], color='b', alpha=0.7, label='Pitch Contour')  # Multiply by 100
        # Set axis limits
        ax.set_ylim(3700, 8400)  # C#2 (3700) to C6 (8400)
        ax.set_xlim(0, round(duration))

        swara_map = {
            # Lower Octave (C#2-B2)
            3700: ('C#2', 'Sa'), 3800: ('D2', 'Ri1'), 3900: ('D#2', 'Ri2'), 4000: ('E2', 'Ga2'),
            4100: ('F2', 'Ga3'), 4200: ('F#2', 'Ma1'), 4300: ('G2', 'Ma2'), 4400: ('G#2', 'Pa'),
            4500: ('A2', 'Da1'), 4600: ('A#2', 'Da2'), 4700: ('B2', 'Ni2'), 4800: ('C3', 'Ni3'),
    
            # Middle Octave (C#3-B3)
            4900: ('C#3', 'Sa'), 5000: ('D3', 'Ri1'), 5100: ('D#3', 'Ri2'), 5200: ('E3', 'Ga2'),
            5300: ('F3', 'Ga3'), 5400: ('F#3', 'Ma1'), 5500: ('G3', 'Ma2'), 5600: ('G#3', 'Pa'),
            5700: ('A3', 'Da1'), 5800: ('A#3', 'Da2'), 5900: ('B3', 'Ni2'), 6000: ('C4', 'Ni3'),
            
            # Upper Octave (C#4-B4)
            6100: ('C#4', 'Sa'), 6200: ('D4', 'Ri1'), 6300: ('D#4', 'Ri2'), 6400: ('E4', 'Ga2'),
            6500: ('F4', 'Ga3'), 6600: ('F#4', 'Ma1'), 6700: ('G4', 'Ma2'), 6800: ('G#4', 'Pa'),
            6900: ('A4', 'Da1'), 7000: ('A#4', 'Da2'), 7100: ('B4', 'Ni2'), 7200: ('C5', 'Ni3'),
            
            # Higher Octaves (C#5-C6)
            7300: ('C#5', 'Sa'),
        }

        # Add horizontal lines and labels for each swara
        for midicent, (note, swara) in swara_map.items():
            ax.axhline(y=midicent, color='gray', linestyle='--', alpha=0.3)
            ax.text(0.5, midicent+20, f'{swara} ({note})',  # Adjusted text position
                    ha='left', va='bottom', color='darkred', fontsize=9,
                    backgroundcolor=(1,1,1,0.7))

        ax2 = ax.twinx()
        mask2 = voiced_prob >= VOICE_PROB_THRESHOLD
        mask2 = mask2 & mask
        print(f'Percentage voiced_prob above : {sum(mask2) * 100.0 / total_masked}')
        ax2.plot(xaxis[mask2], voiced_prob[mask2], color='r', alpha=0.5, label='Voiced Probability')
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('Voiced Probability')

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper left')
        
        # Formatting
        #ax.set_ylabel('MIDI Cents (100× semitone)')
        #ax.set_xlabel('Time (seconds)')
        #ax.title('Carnatic Swara Annotations with C# as Sa')
        ax.grid(True, which='both', axis='y', alpha=0.5)
        ax.set_xticks(np.arange(0, round(duration) + 1, 5))

        # Add octave labels on right side
        # octave_labels = {
        #     3700: 'C#2', 
        #     4900: 'C#3', 
        #     6100: 'C#4', 
        #     7300: 'C#5',
        #     8400: 'C6'
        # }

        # for pos, label in octave_labels.items():
        #     plt.text(30.2, pos, f'{label} Octave', 
        #             rotation=90, va='center', color='darkgreen')

        plt.tight_layout()
        plt.show()
        #with open("tests/test_midi.txt", "w") as f, open("tests/test_stereo.txt", "w") as g:
        #    for i in range(len(pitches1)):


    @pytest.mark.skip(reason="Temporarily disabled for demonstration purposes")
    def test_pitch_change(self):
        RATE = 44100
        FRAME_LENGTH = 2048
        HOP_LENGTH = 512
        audio, sr = librosa.load('../data/simple-test/saveri/tmk-saveri.mp3', sr=44100)
        
        results = {}
        inputs = [
            (-1, '../data/simple-test/saveri/new_audio_minus_1.mp3'),
            (-2, '../data/simple-test/saveri/new_audio_minus_2.mp3'),
            (1, '../data/simple-test/saveri/new_audio_plus_1.mp3'),
            (2, '../data/simple-test/saveri/new_audio_plus_2.mp3')
            ]

        start_time = time.time()
        print(f'Before submission')
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            for i, inp in enumerate(inputs):
                results[executor.submit(self.pitch_shift_helper, audio, sr, inp[0], inp[1])] = inp
                print(f'submitted {i} at {int( (time.time() - start_time) * 1000)}')
        
            print(f'waiting for results at {int( (time.time() - start_time) * 1000)}....')
            for future in concurrent.futures.as_completed(results):
                inp = results[future]
                print(f'written to {inp[1]} after {int( (time.time() - start_time) * 1000)}')

        # new_audio_minus_1 = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-1)
        # output_1 = '../data/simple-test/new_audio_minus_1.mp3'
        # sf.write(output_1, new_audio_minus_1, sr)

        # new_audio_plus_2 = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
        # output_2 = '../data/simple-test/new_audio_plus_2.mp3'
        # sf.write(output_1, new_audio_plus_2, sr)
        


    
    @pytest.mark.skip(reason="Temporarily disabled for demonstration purposes")
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
                note = librosa.hz_to_note(pitches[i]).replace('♯', '#')
                svara = note
                #svara = note_to_svara[note]
                #svara = str(round(pitches[i], 2))
            if vp > 0.5:
                music_pitches.append((timestamp, vp, svara))
        with open("tests/test_mayamalavagowlai_ragasurabhi.txt", "w") as f:
            f.writelines([str(round(item[0], 3)) + ',' + str(round(item[1], 3)) + "," + item[2] + "\n" for item in music_pitches])

    @pytest.mark.skip(reason="Temporarily disabled for demonstration purposes")
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

    @pytest.mark.skip(reason="Temporarily disabled for demonstration purposes")
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
