# MusicVAE - Drummer
Will be closed soon

## 0. Requirements
pretty_midi / pypianoroll

## 1. Preprocess
```bash
$ python preprocess.py
```
[Warning] pypianoroll have a small bug.
* You should fix this small bug to execute preprocess.py
```bash
$ vi {python_packages_directory}/site-package/pypianoroll/inputs.py
```

* line 256 would be like below

```python
        note_ons = rounded.astype(int)
```
* Right after line 256, code below
```python
        while note_ons[-1] == len(pianoroll):
            note_ons = note_ons[:-1]
            pitches = pitches[:-1]
```

* then, find below code (maybe in line 260~270)
```python
        elif instrument.is_drum:
            velocities = [
                note.velocity
                for note in instrument.notes
                if note.end > first_beat_time
            ]
            
            pianoroll[note_ons, pitches] = velocities
```

* Add a line between the declations of velocity and pianoroll[...] like below

```python
        elif instrument.is_drum:
            velocities = [
                note.velocity
                for note in instrument.notes
                if note.end > first_beat_time
            ]
            velocities = velocities[:len(note_ons)]
            pianoroll[note_ons, pitches] = velocities
```
## 2. train
```bash
$ python main.py
```

## 3. inference
```bash
$ python inference.py
```
