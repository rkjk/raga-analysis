import numpy as np
from librosa.feature import chroma_stft

class ChromaDetector:
    def __init__(self, sr, frame_length, hop_length, n_chromas=12):
        self.sr = sr
        self.n_chromas = n_chromas
        self.n_fft = frame_length
        self.hop_length = hop_length
        self.freqs = np.arange(0, self.sr/2, self.sr/self.n_chromas)
        
    def detect(self, signal):
        '''
        Detect the chroma of a given signal.
        :param signal: numpy array of floats
        :return: numpy array of floats
        '''
        return chroma_stft(y=signal, 
            sr=self.sr,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            n_chroma=self.n_chromas)