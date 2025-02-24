import torch
import torch.multiprocessing as mp

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import pprint
import librosa
import numpy as np
import math

from model.cnn1d_adaptive import *
from model.lstm1 import *
from pyin_pitch_detect import *
import utils

NUM_SECONDS = 10
BLOCK_SIZE = 87 * NUM_SECONDS

def infer(model, raga_name, raga_path, device):
    # Training code here
    #print(f'name = {raga_name}, path={raga_path}')
    pitch_detector = PYINPitchDetect(utils.SAMPLE_RATE, frame_length=2048, hop_length=512)
    stoi, itos, _ = utils.get_tokenizer_midi()
    raga_map = utils.get_classes()

    raga_path = os.path.join(raga_path, "source-separated", "vocals")

    def pad_with_not_voice(buf):
        if len(buf) < BLOCK_SIZE:
            for _ in range(BLOCK_SIZE - len(buf)):
                buf.append(stoi[utils.NOT_VOICE_TOKEN])
        return buf

    files = os.listdir(raga_path)
    total_frames = 0
    correct_frames = 0
    incorrect_frames = 0
    for f in files:
        #print(f'Processing file {f}')
        #print(f'Model device: {next(model.parameters()).device}')
        path = os.path.join(raga_path, f)
        if not os.path.isfile(path) or not f.endswith('.mp3'):
            continue
        audio, _ = librosa.load(path,sr=utils.SAMPLE_RATE)
        pitches, voiced_flag, voiced_prob = pitch_detector.detect(audio)
        #print(f'Detected pitches for {f}')
        music_pitches = []
        ######################
        ### Remove NOT_VOICE_TOKEN
        ######################
        #pitches = [p for p in pitches if p != utils.NOT_VOICE_TOKEN]
        for i in range(len(pitches)):
            #svara = utils.NOT_VOICE_TOKEN
            #if voiced_flag[i] and voiced_prob[i] > 0.5:
            #    svara = librosa.hz_to_note(pitches[i]).replace('♯', '#')

            # Change to MIDI
            svara = utils.NOT_VOICE_TOKEN_MIDI
            if voiced_prob[i] >= 0.3 and not math.isnan(pitches[i]):
                svara = int(np.round(librosa.hz_to_midi(pitches[i]) * 100))
            music_pitches.append(svara)
        #music_pitches = pad_with_not_voice(music_pitches)
        music_pitches_new = []
        not_voice_count = 0
        for p in music_pitches:
            if utils.MIN_TOKEN <= p <= utils.MAX_TOKEN:
                not_voice_count = 0
                music_pitches_new.append(p)
            # elif utils.NOT_VOICE_TOKEN_MIDI == p:
                # not_voice_count += 1
                # if not_voice_count == 10:
                #     music_pitches_new.append(utils.NOT_VOICE_TOKEN_MIDI)
                #     not_voice_count = 0
        music_pitches = [stoi[x] for x in music_pitches_new]
        #print(f'Length: {len(music_pitches)} NumFrames: {len(music_pitches) // BLOCK_SIZE}')  
        i = 0
        frame = 0

        while i < len(music_pitches):
            total_frames += 1
            data = None
            if i + BLOCK_SIZE//4 < len(music_pitches):
                data = music_pitches[i:i+BLOCK_SIZE]
            else:
                #data = pad_with_not_voice(music_pitches[i:])
                total_frames -= 1
                break
            i += BLOCK_SIZE // 4
            with torch.no_grad():
                data = torch.tensor(data).view(1,-1).to(device)
                logits = model(data)
                #print(f'{f} -> Completed inference')
                probabilities = torch.softmax(logits, dim=1)
                #print(f'probs: {probabilities}')
                _, max_index = logits.max(dim=1)
                predicted_raga = raga_map[max_index.item()]
                if predicted_raga == raga_name:
                    correct_frames += 1
                else:
                    incorrect_frames += 1        #p = mp.Process(target=infer, args=(model, raga_name, raga))

                #print(f'File: {f} -> Prediction: {predicted_raga}')
            frame += 1
    return total_frames, correct_frames, incorrect_frames

def infer_model(model_path, model, device):
    print(f'Inferring path {model_path}')
    #MODEL_PATH = './models/cnn-1-adam-1e3-epochs-[230000]'
    checkpoint = torch.load(model_path)
    epochs = checkpoint['epochs']
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']

    output = {'ragas': {}}
    output['checkpoint after epoch'] = epochs
    output['train loss'] = train_loss
    output['val loss'] = val_loss
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    #model.share_memory()  # Share the model on CPU
    input_dir = "../data/simple-test/test"
    ragas = get_subdirectories(input_dir)
    #print(ragas)
    for raga in ragas:
        raga_name = os.path.basename(raga)
        if raga_name not in utils.CLASS_NAMES:
            #print(f'{raga_name} not in CLASS_NAMES')
            continue
        t, c, i = infer(model, raga_name, raga, device)
        #print(f'Raga {raga_name} -> total_frames={t}, correct_frames={c}, incorrect={i}, percent_correct={c/t}')
        output['ragas'][raga_name] = {
            'total_frames': t,
            'correct_frames': c,
            'incorrect_frames': i,
            'percent_correct': c/t
        }
    return output
    
        

# TODO: Add to utils get all subdirectories in a given directory
def get_subdirectories(root_dir):
    subdirectories = []
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            subdirectories.append(item_path)
    return subdirectories

if __name__ == '__main__':
    stoi, itos, vocab_size = utils.get_tokenizer_midi()
    #raga_map = utils.get_classes()

    # Model
    in_channels = 1  # Input channels (e.g., single-channel audio)
    out_channels = len(utils.CLASS_NAMES)  # Output channels (e.g., regression output)
    n_tokens = vocab_size

    # CNN
    # kernel_size = 3   # Kernel size
    # stride = 1       # Stride
    # padding = 0      # Padding
    # n_embd = 8

    # LSTM
    hidden_size = 128
    num_layers = 2
    n_embd = 96

    #lr = 0.001
    #epochs = 0
    device = 'cuda:0'
    #device = 'cpu'
    MODEL_PATH = './models/lstm-midi-4-thodi-first'
    epochs = list(range(26, 31, 1))
    futures = {}
    results = {}
    max_workers = 8
    mp.set_start_method('spawn')
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for ep in epochs:
            path = MODEL_PATH + '-epochs-[' + str(ep) + ']'
            # model = ConvNet_1D(in_channels, out_channels, kernel_size, n_embd, n_tokens, device='cuda:0')
            model = LSTMNet(out_channels, n_embd, n_tokens,hidden_size, num_layers, bidirectional=True, dropout=0.3, device=device)
            futures[executor.submit(infer_model, path, model, device)] = path
        
        
        for f in as_completed(futures):
            p = futures[f]
            try:
                result = f.result()
                results[p] = result
                #print(result)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                print(f"Exception occurred on line {exc_tb.tb_lineno}")
                print(f'Exception type: {exc_type}')
                print(f"Exception for {p}: {e}")
    with open("recorded-inference.txt", 'w') as outfile:
        pprint.pprint(results, stream=outfile, indent=4)
