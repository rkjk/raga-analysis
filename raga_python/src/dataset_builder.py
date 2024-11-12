import concurrent.futures
import os
import csv
import argparse
from dataclasses import dataclass
import time

import librosa
from numpy import argmax

from pyin_pitch_detect import *

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
            self.max_workers = os.cpu_count()
        self.executor = self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers)
        print(f'Using {self.max_workers} CPU cores')
        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.metadata = self.read_metadata_csvs(self.get_subdirectories(input_dir))
        self.pyin = PYINPitchDetect(sr, frame_length=self.frame_length, hop_length=self.hop_length)
    
    def get_metadata(self):
        return self.metadata

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

    def process_raga_file_helper(self, audio, output_file_relative_path, dry_run=False):
        try:
            if os.path.isfile(output_file_relative_path) and not self.check_file_empty_or_incomplete(output_file_relative_path):
                return
            pitches, voiced_flag, voiced_prob = self.pyin.detect(audio)
            timestamps = get_timestamps(pitches, HOP_LENGTH, RATE)

            music_pitches = []
            music_probs = []
            for i in range(len(pitches)):
                timestamp = timestamps[i]
                svara = NOT_VOICE_TOKEN
                if voiced_flag[i] and voiced_prob[i] > 0.5:
                    svara = librosa.hz_to_note(pitches[i]).replace('♯', '#')
                music_pitches.append((timestamp, vp, svara))
            
            print(f'Writing to {output_file_relative_path}')
            if not dry_run:
                with open(output_file_relative_path, 'w'):
                    f.writelines([str(round(item[0], 3)) + ',' + str(round(item[1], 3)) + "," + item[2] + "\n" for item in music_pitches])
                    f.write(END_OF_FILE_TOKEN)
                    print(f'SUCCESS: Written to {output_file_relative_path}')
            return True
        except Exception as ex:
            err = f'Error processing helper file {output_file_relative_path}: {ex}'
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

            # Main track
            output_file_path = os.path.join(output_file_dir, f_name)
            results[self.executor.submit(self.process_raga_file_helper, audio, output_file_path, dry_run)] = output_file_path

            #Shifted Tonics
            for n_steps in shift_tonics:
                shifted = librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=n_steps)
                n_steps_abs = abs(n_steps)
                sf_name = '_'.join([f_name, "minus", str(n_steps_abs)]) if n_steps < 0 else '_'.join([f_name, "plus", str(n_steps_abs)])
                sf_output_file_path = os.path.join(output_file_dir, sf_name)
                results[self.executor.submit(self.process_raga_file_helper, shifted, sf_output_file_path, dry_run)]
            
            for future in concurrent.futures.as_completed(results):
                res = future.result()
            print(f'Completed {raga_file_data.relative_path} in {int( (time.time() - start_time) * 1000)} seconds')
        except Exception as ex:
            err = f'Error processing file {raga_file_data.relative_path}: {ex}'
            raise RuntimeError(err)

    def process_all(self):
        saveri_raga_files = self.metadata['saveri']
        for f in saveri_raga_files:
            self.process_raga_file(f, dry_run=True)        


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
    args = parser.parse_args()
    if args.input_dir is None:
        err = f'input_dir param is mandatory'
        raise Exception(err)
    if args.output_dir is None:
        err = f'output_dir param is mandatory'
        raise Exception(err)
    dataset_builder = ConcurrentDatasetBuilder(args.input_dir, args.output_dir, max_workers=args.max_workers)
    #print(f'{dataset_builder.get_metadata()}')
    dataset_builder.process_all()