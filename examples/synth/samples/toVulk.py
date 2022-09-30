import os
import json 
import librosa
import soundfile as sf
from tqdm import tqdm

NOTES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
NOTES = [n.lower() for n in NOTES]
OCTAVES = list(range(11))
NOTES_IN_OCTAVE = len(NOTES)

def number_to_note(number: int) -> tuple:
    octave = number // NOTES_IN_OCTAVE
    assert octave in OCTAVES, errors['notes']
    assert 0 <= number <= 127, errors['notes']
    note = NOTES[number % NOTES_IN_OCTAVE]

    return note, octave


def note_to_number(note: str, octave: int) -> int:

    note = NOTES.index(note)
    note += (NOTES_IN_OCTAVE * octave)

    assert 0 <= note <= 127, errors['notes']

    return note

sampleFilenames = [""]*128

for filename in os.listdir("notes"):
    if filename.endswith("_mf_rr1.wav"):
        midiNoteName = filename.split("_mf_rr1.wav")[0]
        print(midiNoteName)
        octave = int(midiNoteName[-1])
        midiNoteName = midiNoteName[:-1]
        midiNoteNumber = note_to_number(midiNoteName, octave)
        sampleFilenames[midiNoteNumber] = filename
        
distanceToSample = [0]*128
# find nearest non-x
lastNonX = -1
for i in range(128):
    nextNonX = -1
    if sampleFilenames[i] != "":
        lastNonX = i
    for j in range(i, 128):
        if sampleFilenames[j] != "":
            nextNonX = j
            break
            
    
    if lastNonX == -1:
        sampleFilenames[i] = sampleFilenames[nextNonX]
        distanceToSample[i] = i-nextNonX
    elif nextNonX == -1:
        sampleFilenames[i] = sampleFilenames[lastNonX]
        distanceToSample[i] = i-lastNonX
    elif abs(i-lastNonX) < abs(i-nextNonX):
        sampleFilenames[i] = sampleFilenames[lastNonX]
        distanceToSample[i] = i-lastNonX
    else:
        sampleFilenames[i] = sampleFilenames[nextNonX]
        distanceToSample[i] = i-nextNonX
        
        
print(json.dumps(sampleFilenames, indent=2))
print(json.dumps(distanceToSample, indent=2))
skipval = 4
for noteno in tqdm(range(128)):
    y, samplerate = librosa.load(os.path.join("notes", sampleFilenames[noteno]), sr=44100 * skipval)
    y_shifted = librosa.effects.pitch_shift(y, sr=samplerate, n_steps=distanceToSample[noteno])
    sf.write(
        "guitar/midi" + str(noteno) + ".wav", y_shifted, samplerate, subtype="Float"
    )
    
