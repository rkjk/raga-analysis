from src.pyin_pitch_detect import *
from src.chroma_detect import *
from src.utils import *
import src.music_detection_utils as music_detection_utils

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
        FRAME_LENGTH = 1024
        HOP_LENGTH = 256
        duration = 23.0
        audio, sr = librosa.load('../data/separated/htdemucs/raga-adum-chidambaramo-23sec/vocals.wav', sr=44100, duration=duration)
        #audio, sr = librosa.load('../data/raga-adum-chidambaramo-23sec.mp3', sr=44100, duration=duration)
        pyin = PYINPitchDetect(sr, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        pitches, voiced_flag, voiced_prob = pyin.detect(audio)
        timestamps = get_timestamps(pitches, HOP_LENGTH, RATE)
        sound_pitches = []
        unvoice_prob = []
        NAN = librosa.midi_to_hz(21)
        VOICE_PROB_THRESHOLD = 0.3
        
        # Only keep pitches where probability is above threshold
        sound_pitches = [pitches[i] if voiced_prob[i] > VOICE_PROB_THRESHOLD and not math.isnan(pitches[i])
                        else NAN for i in range(len(pitches))]
                
        midi = librosa.hz_to_midi(sound_pitches)
        midi_cents = np.round([m * 100 for m in midi])
        
        # Create combined mask for both pitch range and voice probability
        mask = (midi_cents > 3700) & (voiced_prob >= VOICE_PROB_THRESHOLD)
        total_masked = sum(mask)
        print(f'Number of masked={sum(mask)}')
        
        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        xaxis = np.linspace(0, duration, len(midi_cents))
        
        # Only plot points that meet both criteria
        valid_mask = (midi_cents > 3700) & (voiced_prob >= VOICE_PROB_THRESHOLD)
        ax.scatter(xaxis[valid_mask], 
                midi_cents[valid_mask],
                color='b',
                alpha=0.7,
                s=10,
                label='Pitch Points')
        
        # Set axis limits
        ax.set_ylim(3700, 8400)
        yticks = np.arange(3700, 8401, 100)
        carnatic_svaras_in_order = ["S", "R1", "R2/G1", "R3/G2", "G3", "M1", "M2", "P", "D1", "D2/N1", "D3/N2", "N3"]

        start_tick = 4300  # Starting point for labeling -  G2 (G3 is RaGa's tonic here)
        start_index = np.where(yticks == start_tick)[0][0]  # Find the index of the starting tick

        yticklabels = []
        label_index = 0

        for tick in yticks:
            if tick >= start_tick:
                yticklabels.append(carnatic_svaras_in_order[label_index % len(carnatic_svaras_in_order)])  # Circular labeling
                label_index += 1
            else:
                yticklabels.append('')
        ax.set_yticks(yticks, yticklabels)
        ax.set_xlim(0, round(duration))
        
        
        lines, labels = ax.get_legend_handles_labels()
        ax.legend(lines, labels, loc='upper left')
        
        ax.grid(True, which='both', axis='y', alpha=0.5)
        ax.set_xticks(np.arange(0, round(duration) + 1, 5))
        
        plt.tight_layout()
        plt.show()


    @pytest.mark.skip(reason="Temporarily disabled for demonstration purposes")
    def test_music_detection_rule_based(self):
        RATE = 44100
        FRAME_LENGTH = 2048
        HOP_LENGTH = 512

        music_percentage, speech_ranges, music_ranges = utils.detect_speech_music(
            '../data/raga_comparison-saveri_and_malahari.mp3',
            segment_duration=10,
            output_path='tests/raga-comp-saveri-malahari-music-only.mp3'
            )
        print(f'Music Percentage: {music_percentage}')
        print(f'Music: {music_ranges}')
        print(f'Speech: {speech_ranges}')
        music_percentage, speech_ranges, music_ranges = utils.detect_speech_music(
            '../data/visweswaran-abhogi-class.mp3',
            segment_duration=10,
            output_path='tests/visweswaran-abhogi-class-music-only.mp3'
            )
        print(f'Music Percentage: {music_percentage}')
        print(f'Music: {music_ranges}')
        print(f'Speech: {speech_ranges}')
        music_percentage, speech_ranges, music_ranges = utils.detect_speech_music(
            '../data/sindhubhairavi-chari-sisters.mp3',
            segment_duration=10,
            output_path='tests/sindhubhairavi-chari-sisters.mp3-music-only.mp3'
            )
        print(f'Music Percentage: {music_percentage}')
        print(f'Music: {music_ranges}')
        print(f'Speech: {speech_ranges}')
        music_percentage, speech_ranges, music_ranges = utils.detect_speech_music(
            '../data/simple-test/input/saveri/arunasairam-muruga-muruga.mp3',
            segment_duration=10,
            output_path='tests/arunasairam-muruga-muruga-music-only.mp3'
            )
        print(f'Music Percentage: {music_percentage}')
        print(f'Music: {music_ranges}')
        print(f'Speech: {speech_ranges}')

    @pytest.mark.skip(reason="Temporarily disabled for demonstration purposes")
    def test_music_detection_pann(self):
        RATE = 44100
        FRAME_LENGTH = 2048
        HOP_LENGTH = 512

        music_percentage, speech_ranges, music_ranges = music_detection_utils.detect_speech_music(
            '../data/raga_comparison-saveri_and_malahari.mp3',
            segment_duration=10,
            output_path='tests/raga-comp-saveri-malahari-music-only.mp3'
            )
        print(f'Music Percentage: {music_percentage}')
        print(f'Music: {music_ranges}')
        print(f'Speech: {speech_ranges}')
        music_percentage, speech_ranges, music_ranges = music_detection_utils.detect_speech_music(
            '../data/simple-test/thodi-vittal-rangan.mp3',
            segment_duration=10,
            output_path='tests/thodi-vittal-rangan-music-only.mp3'
            )
        print(f'Music Percentage: {music_percentage}')
        print(f'Music: {music_ranges}')
        print(f'Speech: {speech_ranges}')
        music_percentage, speech_ranges, music_ranges = music_detection_utils.detect_speech_music(
            '../data/sai-giridhar-mridangam.wav',
            segment_duration=10,
            output_path='tests/sai-giridhar-mridangam-music-only.mp3'
            )
        print(f'Music Percentage: {music_percentage}')
        print(f'Music: {music_ranges}')
        print(f'Speech: {speech_ranges}')
        music_percentage, speech_ranges, music_ranges = music_detection_utils.detect_speech_music(
            '../data/visweswaran-abhogi-class.mp3',
            segment_duration=10,
            output_path='tests/visweswaran-abhogi-class-music-only.mp3'
            )
        print(f'Music Percentage: {music_percentage}')
        print(f'Music: {music_ranges}')
        print(f'Speech: {speech_ranges}')
        music_percentage, speech_ranges, music_ranges = music_detection_utils.detect_speech_music(
            '../data/sindhubhairavi-chari-sisters.mp3',
            segment_duration=10,
            output_path='tests/sindhubhairavi-chari-sisters.mp3-music-only.mp3'
            )
        print(f'Music Percentage: {music_percentage}')
        print(f'Music: {music_ranges}')
        print(f'Speech: {speech_ranges}')
        music_percentage, speech_ranges, music_ranges = music_detection_utils.detect_speech_music(
            '../data/simple-test/input/saveri/arunasairam-muruga-muruga.mp3',
            segment_duration=10,
            output_path='tests/arunasairam-muruga-muruga-music-only.mp3'
            )
        print(f'Music Percentage: {music_percentage}')
        print(f'Music: {music_ranges}')
        print(f'Speech: {speech_ranges}')


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
