import torch
import torch.multiprocessing as mp

import os
import librosa

from model.cnn1d import *
from pyin_pitch_detect import *
import utils

def infer(model, raga_name, raga_path, device):
    # Training code here
    #print(f'name = {raga_name}, path={raga_path}')
    pitch_detector = PYINPitchDetect(utils.SAMPLE_RATE, frame_length=2048, hop_length=512)
    stoi, itos, _ = utils.get_tokenizer()

    def pad_with_not_voice(buf):
        if len(buf) < 2700:
            for _ in range(2700 - len(buf)):
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
        audio, _ = librosa.load(path,sr=utils.SAMPLE_RATE)
        pitches, voiced_flag, voiced_prob = pitch_detector.detect(audio)
        #print(f'Detected pitches for {f}')
        music_pitches = []
        for i in range(len(pitches)):
            svara = utils.NOT_VOICE_TOKEN
            if voiced_flag[i] and voiced_prob[i] > 0.5:
                svara = librosa.hz_to_note(pitches[i]).replace('♯', '#')
            music_pitches.append(svara)
        music_pitches = pad_with_not_voice(music_pitches)
        music_pitches = [stoi[x] for x in music_pitches]
        #print(f'Length: {len(music_pitches)} NumFrames: {len(music_pitches) // 2700}')  
        i = 0
        frame = 0

        while i < len(music_pitches):
            total_frames += 1
            data = None
            if i + 2700 < len(music_pitches):
                data = music_pitches[i:i+2700]
            else:
                data = pad_with_not_voice(music_pitches[i:])
            i += 2700
            not_voice_count = data.count(0)
            if not_voice_count >= 2600:
                #print(f'File: {f} -> Prediction: NOT_VOICE. Count {not_voice_count}')
                correct_frames += 1
                continue
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
                    incorrect_frames += 1
                #print(f'File: {f} -> Prediction: {predicted_raga}')
            frame += 1
    return total_frames, correct_frames, incorrect_frames
    
        

# TODO: Add to utils get all subdirectories in a given directory
def get_subdirectories(root_dir):
    subdirectories = []
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            subdirectories.append(item_path)
    return subdirectories

if __name__ == '__main__':
    stoi, itos, vocab_size = utils.get_tokenizer()
    raga_map = utils.get_classes()

    # Model
    in_channels = 1  # Input channels (e.g., single-channel audio)
    out_channels = 2  # Output channels (e.g., regression output)
    kernel_size = 3   # Kernel size
    stride = 1       # Stride
    padding = 0      # Padding
    n_tokens = vocab_size
    n_embd = 24

    lr = 0.001
    epochs = 0
    device = 'cuda:0'
    model = ConvNet_1D(in_channels, out_channels, kernel_size, n_embd, n_tokens, device='cuda:0')
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
    #model.share_memory()  # Share the model on CPU

    processes = []
    input_dir = "../data/simple-test/ragasurabhi"
    ragas = get_subdirectories(input_dir)
    print(ragas)
    for raga in ragas:
        raga_name = os.path.basename(raga)
        if raga_name not in utils.CLASS_NAMES:
            print(f'{raga_name} not in CLASS_NAMES')
            continue
        #p = mp.Process(target=infer, args=(model, raga_name, raga))
        t, c, i = infer(model, raga_name, raga, device)
        print(f'Raga {raga_name} -> total_frames={t}, correct_frames={c}, incorrect={i}, percent_correct={c/t}')
        #p.start()
        #processes.append(p)

    #for p in processes:
    #    p.join()
