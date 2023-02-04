import numpy as np
import pretty_midi as pm

_eye = np.eye(128, dtype=bool)
_tempo = 120

def _profile_to_binary(profile):
    binary = bin(profile)[2:]
    binary = "0" * (9-len(binary)) + binary
    return binary

def _get_binary_to_indices(binary):
    indices = [ROLAND_DRUM_PITCH_CLASSES[i] for i, b in enumerate(binary) if b == "1"]
    return indices

def _profile_to_pianoroll(profile):
    binary = _profile_to_binary(profile)
    indices = _get_binary_to_indices(binary)
    return _eye[indices].sum(0)
    
def _pr_to_midi(pr, bpm=_tempo, beat_res=4):
    unit_time = 60 / (bpm * beat_res)
    inst = pm.Instrument(program=0, is_drum=True)
    locs, pitches = np.where(pr)
    for loc, pitch in zip(locs, pitches):
        start_t, end_t = loc * unit_time, (loc+1) * unit_time
        note = pm.Note(start=start_t, end=end_t, pitch=pitch, velocity=60)
        inst.notes.append(note)
    
    return inst


ROLAND_DRUM_PITCH_CLASSES = [
    # kick drum
    36, # 256
    # snare drum
    38, # 128
    # closed hi-hat
    42, # 64
    # open hi-hat
    46, # 32
    # low tom
    43, # 16
    # mid tom
    47, # 8
    # high tom
    50, # 4
    # crash cymbal
    49, # 2
    # ride cymbal
    51 # 1
]

def _get_pianoroll_from_profile(profiles):
    return np.stack([_profile_to_pianoroll(p) for p in profiles])

def get_midi_from_profile(profiles):
    pianoroll = _get_pianoroll_from_profile(profiles)
    inst = _pr_to_midi(pianoroll)
    midi = pm.PrettyMIDI(initial_tempo=_tempo)
    midi.instruments.append(inst)
    
    return midi