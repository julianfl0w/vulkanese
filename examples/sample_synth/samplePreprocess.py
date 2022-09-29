# Extract harmonic and percussive components
import numpy as np
import librosa
import soundfile as sf
y, samplerate = librosa.load('guitarpluck_467672__allan764__21-c4.wav')

y_harmonic, y_percussive = librosa.effects.hpss(y)
for i in (np.arange(128) - 60): # where 60 is C4. this makes each i a MIDI note
    y_shifted = librosa.effects.pitch_shift(y_harmonic, sr=samplerate, n_steps=i)
    sf.write("guitar/midi" + str(i+60) + ".wav", y_shifted + y_percussive, samplerate, subtype='Float')
    #print(np.shape(y_harmonic))