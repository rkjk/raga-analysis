import torch
import torch.multiprocessing as mp

from model.cnn1d import *
import utils

def infer(model, process_id):
    # Training code here
    print(f'process=id {process_id}')

if __name__ == '__main__':
    num_processes = 4

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
    model = ConvNet_1D(in_channels, out_channels, kernel_size, n_embd, n_tokens, device='cpu')
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
    model.share_memory()  # Share the model on CPU

    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=infer, args=(model, rank))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
