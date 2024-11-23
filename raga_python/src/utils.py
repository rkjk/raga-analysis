from numpy import arange

SAMPLE_RATE=44100
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOT_VOICE_TOKEN = '<N>'
END_OF_FILE_TOKEN = '<EOF>'
BLOCK_SIZE = 87 * 30 + 90
# CLASS NAMES - NOTE: Do not change numbering.
CLASS_NAMES = {
    'saveri': 0,
    'hemavati': 1
}

def get_current_time_microseconds():
    return int(time.time() * 1e6)

def decompose_note(note: str):
    # Get octave and base note
    if not note:
        return None, None
    octave = int(note[-1]) if note[-1].isdigit() else None
    base = note[:-1] if note[-1].isdigit() else note
    return base, octave

def is_complete_octave(bag: dict):
    # Do we have 7 notes in one octave and one note in the next
    if not bag:
        return False, None
    octaves_present_in_bag = list(sorted(bag.keys()))
    for oct in octaves_present_in_bag[:-1]:
        if len(bag[oct]) >= 7 and len(bag[oct + 1]) >= 1:
            return True, oct
    return False, None

# Calculate the starting timestamp of each frame for which pitch is computed
def get_timestamps(pitches: list, hop_length: int, sr: int) -> list:
    num_frames =  len(pitches)
    return arange(num_frames) * hop_length * 1.0 / sr

def get_tokenizer():
    # Tokenizer
    ALLOWED_TOKENS = []
    for octave in [2,3,4,5]:
        o = str(octave)
        for n in NOTES:
            ALLOWED_TOKENS.append(n+o)

    stoi = {s:i+1 for i,s in enumerate(ALLOWED_TOKENS)}
    stoi[NOT_VOICE_TOKEN] = 0
    itos = {i:s for s,i in stoi.items()}
    vocab_size = len(itos)
    return stoi, itos, vocab_size

def get_classes():
    reverse_map = {}
    for k,v in CLASS_NAMES.items():
        if v in reverse_map:
            raise RuntimeError(f'Duplicate key found {k} -> {v} -> {reverse_map[v]}')
        reverse_map[v] = k
    return reverse_map