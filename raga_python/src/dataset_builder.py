import concurrent.futures
import os
import csv
import argparse
from dataclasses import dataclass

@dataclass
class RagaFileData:
    relative_path: str
    raga_name: str
    file_name: str
    tonic: str
    url: str

class ConcurrentDatasetBuilder:
    def __init__(self, input_dir, output_dir, max_workers=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.executor = None
        if max_workers is not None:
            self.max_workers = max_workers
        else:
            self.max_workers = os.cpu_count()
        print(f'Using {self.max_workers} CPU cores')
        self.metadata = self.read_metadata_csvs(self.get_subdirectories(input_dir))
    
    def get_metadata(self):
        return self.metadata

    def process_all(self):
        if self.executor == None:
            self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers)

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
                        file_name = row[0]
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
    print(f'{dataset_builder.get_metadata()}')