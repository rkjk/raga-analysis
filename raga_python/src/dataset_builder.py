import concurrent.futures
import os
import csv
import argparse
from dataclasses import dataclass
import time

import librosa
from numpy import argmax, arange

#from pyin_pitch_detect import *

# Calculate the starting timestamp of each frame for which pitch is computed
def get_timestamps(pitches: list, hop_length: int, sr: int) -> list:
    num_frames =  len(pitches)
    return arange(num_frames) * hop_length * 1.0 / sr

class PYINPitchDetect:
    def __init__(self, sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), frame_length=None, hop_length=None):
        if not sr:
            raise RuntimeError("Sample rate not specified")
        if not frame_length:
            self.frame_length = 1024
        self.sr = sr
        self.fmin = fmin
        self.fmax = fmax
        self.frame_length = frame_length
        self.hop_length = frame_length // 2 if not hop_length else hop_length
        print(f'Created PYINPitchDetect with sr={sr}, fmin={self.fmin}, fl={self.frame_length}, hl={self.hop_length}')
    
    def detect(self, data):
        try:
            return librosa.pyin(
                    data,
                    fmin=self.fmin, 
                    fmax=self.fmax,
                    sr=self.sr,
                    frame_length=self.frame_length,
                    hop_length=self.hop_length)
        except Exception as ex:
            print(f'Error: PYIN -> {ex}')
            raise RuntimeError(ex)
    
    def name(self):
        return "PYIN"

@dataclass
class RagaFileData:
    relative_path: str
    raga_name: str
    file_name: str
    tonic: str
    url: str

ALLOWED_TONICS = ['C3', 'C#3', 'D3', 'D#3', 'E3', 'F3', 'F#3', 'G3', 'G#3', 'A3', 'A#3', 'B3']
NOT_VOICE_TOKEN = '<N>'
END_OF_FILE_TOKEN = '<EOF>'

class ConcurrentDatasetBuilder:
    def __init__(self, 
            input_dir, 
            output_dir, 
            max_workers=None, 
            sr=44100, 
            frame_length=2048, 
            hop_length=512):
        self.input_dir = input_dir
        self.output_dir = output_dir
        if max_workers is not None:
            self.max_workers = max_workers
        else:
            self.max_workers = 8
        #self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers)
        print(f'Using {self.max_workers} CPU cores')
        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.metadata = self.read_metadata_csvs(self.get_subdirectories(input_dir))
    
    def get_metadata(self):
        return self.metadata

    @staticmethod
    def check_file_empty_or_incomplete(relative_path: str):
        if not os.path.isfile(relative_path):
            print(f'File does not exist')
            return False
        if os.path.getsize(relative_path) == 0:
            print(f'File {relative_path} is empty')
            return True
        with open(relative_path, 'r') as file:
            # Seek to the end of the file and then back up to the last newline character
            file.seek(0, 2)  # Seek to the end of the file
            pos = file.tell()  # Get the current position (file size)
            file.seek(pos - 1, 0)  # Go back one character
            while file.read(1) != '\n':  # Back up until we find a newline character
                file.seek(file.tell() - 2, 0)
            last_line = file.readline().strip()
            if last_line != END_OF_FILE_TOKEN:
                print(f'File {relative_path} is incomplete')
                return True
        return False
    
    @staticmethod
    def check_file_exists(output_file_relative_path: str):
        if os.path.isfile(output_file_relative_path) and not ConcurrentDatasetBuilder.check_file_empty_or_incomplete(output_file_relative_path):
            return True
        return False

    @staticmethod
    def process_raga_file_helper(audio, output_file_relative_path, dry_run=False):
        try:
            print(f'start process for {output_file_relative_path}')
            if os.path.isfile(output_file_relative_path) and not ConcurrentDatasetBuilder.check_file_empty_or_incomplete(output_file_relative_path):
                return
            #print(f'Starting pitch detection for {output_file_relative_path}')
            pitches, voiced_flag, voiced_prob = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('B5'),
                sr=44100,
                frame_length=2048,
                hop_length=512
            )
            print(f'obtained pitches for {output_file_relative_path}')
            timestamps = get_timestamps(pitches, 512, 44100)

            music_pitches = []
            music_probs = []
            for i in range(len(pitches)):
                timestamp = timestamps[i]
                svara = NOT_VOICE_TOKEN
                if voiced_flag[i] and voiced_prob[i] > 0.5:
                    svara = librosa.hz_to_note(pitches[i]).replace('♯', '#')
                music_pitches.append((timestamp, voiced_prob[i], svara))
            
            #print(f'Writing to {output_file_relative_path}')
            if not dry_run:
                directory_path = os.path.dirname(output_file_relative_path)
                os.makedirs(directory_path, exist_ok=True)
                with open(output_file_relative_path, 'w') as f:
                    f.writelines([str(round(item[0], 3)) + ',' + str(round(item[1], 3)) + "," + item[2] + "\n" for item in music_pitches])
                    f.write(END_OF_FILE_TOKEN)
                    print(f'SUCCESS: Written to {output_file_relative_path}')
            return True
        except Exception as ex:
            err = f'Error processing helper file {output_file_relative_path}: {ex}'
            print(err)
            return False

    def process_raga_file(self, raga_file_data: RagaFileData, dry_run=False):
        try:
            start_time = time.time()
            audio, _ = librosa.load(raga_file_data.relative_path, sr=self.sr)
            tonic_index = ALLOWED_TONICS.index(raga_file_data.tonic)
            shift_tonics = []
            # Possible shift tonics - we use the semitones upto 3 higher amd 2 lower than current
            # As long as we don't go beyond B3 or below C3.
            for t in [-2,-1,1,2,3]:
                if tonic_index + t >= 0 and tonic_index + t < len(ALLOWED_TONICS):
                    shift_tonics.append(t)

            f_name = '_'.join([raga_file_data.file_name, raga_file_data.tonic])
            output_file_dir = os.path.join(self.output_dir, raga_file_data.raga_name)
            results = {}
            #pyin = PYINPitchDetect(self.sr, frame_length=self.frame_length, hop_length=self.hop_length)

            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Main track
                output_file_path = os.path.join(output_file_dir, f_name)
                if not self.check_file_exists(output_file_path):
                    results[executor.submit(ConcurrentDatasetBuilder.process_raga_file_helper, audio, output_file_path, dry_run)] = output_file_path
                else:
                    print(f'Skipping {output_file_path}')

                #Shifted Tonics
                for n_steps in shift_tonics:
                    n_steps_abs = abs(n_steps)
                    sf_name = '_'.join([f_name, "minus", str(n_steps_abs)]) if n_steps < 0 else '_'.join([f_name, "plus", str(n_steps_abs)])
                    sf_output_file_path = os.path.join(output_file_dir, sf_name)
                    if self.check_file_exists(sf_output_file_path):
                        print(f'Skipping {sf_output_file_path}')
                        continue
                    shifted = librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=n_steps)
                    results[executor.submit(ConcurrentDatasetBuilder.process_raga_file_helper, shifted, sf_output_file_path, dry_run)] = sf_output_file_path
                for future in concurrent.futures.as_completed(results):
                    pass
            print(f'Completed {raga_file_data.relative_path} in {int( (time.time() - start_time) * 1000)} ms')
        except Exception as ex:
            err = f'Error processing file {raga_file_data.relative_path}: {ex}'
            raise RuntimeError(err)

    def process_all(self, dry_run):       
        for k, files in self.metadata.items():
            for f in files:
                self.process_raga_file(f, dry_run=dry_run)


    # get all subdirectories in a given directory
    def get_subdirectories(self, root_dir):
        subdirectories = []
        for item in os.listdir(root_dir):
            item_path = os.path.join(root_dir, item)
            if os.path.isdir(item_path):
                subdirectories.append(item_path)
        return subdirectories

    # Fetch metadata for each raga
    def read_metadata_csvs(self, subdirectories):
        result = {}
        for subdirectory in subdirectories:
            metadata_path = os.path.join(subdirectory, 'metadata.csv')
            raga_name = os.path.basename(subdirectory)
            if not os.path.exists(metadata_path):
                print(f"Warn: No metadata.csv found in {subdirectory}.") 
                continue
            try:
                with open(metadata_path, mode='r') as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    rows = list(csv_reader)
                    res = []
                    if not rows:
                        print(f"Warn: metadata.csv in {subdirectory} is empty. Skipping.")
                        continue
                    for row in rows:
                        if len(row) < 3:
                            print(f'Warn: Data missing for {relative_path}. Skipping.')
                            continue
                        relative_path = os.path.join(subdirectory, row[0])
                        file_name = ''.join(row[0].split('.')[:-1])
                        tonic = row[1]
                        url = row[2]
                        if tonic is None:
                            print(f'Warn: Tonic missing for {relative_path}. Skipping.')
                            continue
                        res.append(RagaFileData(
                            relative_path=relative_path,
                            file_name=file_name,
                            raga_name=raga_name,
                            tonic=tonic,
                            url=url
                        ))
                    result[raga_name] = res
            except Exception as e:
                print(f"Error parsing metadata.csv in {subdirectory}: {e}")  
        return result              

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Concurrent Dataset Builder')
    parser.add_argument('--input_dir', help='Location containing subdirectories for each raga with data')
    parser.add_argument('--output_dir', help='Location where pitch data will be written')
    parser.add_argument('--max_workers', help='Number of worker. Default is 16')
    parser.add_argument('--dry_run', help='Number of worker. Default is False')
    args = parser.parse_args()
    if args.input_dir is None:
        err = f'input_dir param is mandatory'
        raise Exception(err)
    if args.output_dir is None:
        err = f'output_dir param is mandatory'
        raise Exception(err)
    dry_run = args.dry_run if args.dry_run is not None else False
    dataset_builder = ConcurrentDatasetBuilder(args.input_dir, args.output_dir, max_workers=args.max_workers)
    #print(f'{dataset_builder.get_metadata()}')
    dataset_builder.process_all(dry_run)