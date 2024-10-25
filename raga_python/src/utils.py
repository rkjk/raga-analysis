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