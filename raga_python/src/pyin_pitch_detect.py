from librosa import pyin, note_to_hz

class PYINPitchDetect:
    def __init__(self, sr, fmin=note_to_hz('C2'), fmax=note_to_hz('C7'), frame_length=None, hop_length=None):
        if not sr:
            raise RuntimeError("Sample rate not specified")
        if not frame_length:
            self.frame_length = 1024
        self.sr = sr
        self.fmin = fmin
        self.fmax = fmax
        self.frame_length = frame_length
        self.hop_length = frame_length // 2 if not hop_length else hop_length
    
    def detect(self, data):
        return pyin(
                data,
                fmin=self.fmin, 
                fmax=self.fmax,
                sr=self.sr,
                frame_length=self.frame_length,
                hop_length=self.hop_length)
    
    def name(self):
        return "PYIN"