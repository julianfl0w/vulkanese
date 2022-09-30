# Extract harmonic and percussive components
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import sys

filename = sys.argv[1]
noteno = int(sys.argv[2])
skipval = 4  # GPU Nonsense
y, samplerate = librosa.load(filename, sr=44100 * skipval)
print(samplerate)
y_harmonic, y_percussive = librosa.effects.hpss(y)
for i in tqdm(np.arange(128) - noteno):  # where 60 is C4. this makes each i a MIDI note
    y_shifted = librosa.effects.pitch_shift(y_harmonic, sr=samplerate, n_steps=i)
    sf.write(
        "guitar/midi" + str(i + noteno) + ".wav", y_shifted, samplerate, subtype="Float"
    )
    # print(np.shape(y_harmonic))

sf.write("guitar/midipercussive.wav", y_percussive, samplerate, subtype="Float")
