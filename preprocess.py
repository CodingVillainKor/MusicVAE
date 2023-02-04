from pretty_midi import PrettyMIDI
import pypianoroll
from glob import glob
import os
import pickle as pkl

def main():
    data_root_dir = "/workspace/hrim/groove"
    data_list = []
    for parent, folders, files in os.walk(data_root_dir):
        for file in files:
            if "mid" in file[-4:].lower() and "4-4" in file: # Appendix A. 1. 2 (4/4 TimeSignature)
                data_list.append(os.path.join(parent, file))
    print("len(data_list) = {:,}".format(len(data_list)))

    ROLAND_DRUM_PITCH_CLASSES = [
        # kick drum
        [36], # 256
        # snare drum
        [38, 37, 40], # 128
        # closed hi-hat
        [42, 22, 44], # 64
        # open hi-hat
        [46, 26], # 32
        # low tom
        [43, 58], # 16
        # mid tom
        [47, 45], # 8
        # high tom
        [50, 48], # 4
        # crash cymbal
        [49, 52, 55, 57], # 2
        # ride cymbal
        [51, 53, 59] # 1
    ]

    def get_drum_profiles_from_pianoroll(pr):
        profiles = []
        for pr_t in pr:
            is_hits = []
            for pc in ROLAND_DRUM_PITCH_CLASSES:
                is_hit = str(int(pr_t[pc].any())) # hit: 1 / no-hit: 0
                is_hits.append(is_hit)
            binary_profile = "0b" + "".join(is_hits)
            profiles.append(int(binary_profile, 2)) # 0b101000000 -> 256 + 64 = 320

        return profiles

    dataset = []
    beats = []

    for i, data in enumerate(data_list, 1):
        pm = PrettyMIDI(data)
        beats.append(pm.get_beats())
        pianoroll = pypianoroll.from_pretty_midi(pm, resolution=4).tracks[0].binarize()
        profile_seq = get_drum_profiles_from_pianoroll(pianoroll)
        dataset.append(profile_seq)
        print(f"\r{i} / {len(data_list):,}", end="", flush=True)

    dataset_four_bar = []

    def extract_four_bar(data):
        i = 0 
        datas = []
        while len(data[64*i:64*(i+4)]) == 256:
            datas.append(data[64*i:64*(i+4)])
            i += 1
        return datas
        
    for data in filter(lambda x: len(x) >= 256, dataset): # filter out data of length less than 4 bar
        datas = extract_four_bar(data)
        dataset_four_bar.extend(datas)

    def empty_2bar(data):
        data_m1 = [i-1 for i in data]
        return "-1"*32 not in "".join(map(str, data_m1)) #  Appendix A. 2. 1 (at most a single bar of consecutive rests)

    dataset_four_bar_filtered = list(filter(empty_2bar, dataset_four_bar))
    with open("test.pkl", "wb") as fw:
        pkl.dump(dataset_four_bar_filtered, fw)
        
if __name__ == "__main__":
    main()