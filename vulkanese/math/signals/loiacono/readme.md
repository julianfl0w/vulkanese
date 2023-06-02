# The Loiacono Transform

## Definition

The Loiacono Transform is defined as

$$L_{mag}(f', x, m) = |x \cdot T_r(f', m), x \cdot T_i(f', m)| $$

where x is the signal, m is how many periods of each frequency are considered

the phase angle is ignored because it's not important for my uses

$$
\begin{equation}
T_r(f',m) = 
    \begin{array}{lr}
        \cos({2\pi f' n}), & \text{if } n < p'm \\
        0, & otherwise
    \end{array}
\end{equation}
$$

$$
\begin{equation}
T_i(f',m) = 
    \begin{array}{lr}
        -\sin ({2\pi f' n}), & \text{if } n < p'm \\
        0, & otherwise
    \end{array}
\end{equation}
$$

where p' = 1/f'

for n = 0,1,2...len(x)

## Introduction

The Loiacono Transform makes the following improvements on the Discrete Fourier Transform:
1. An arbitrary list of frequencies can be measured. In DFT, you get all integer multiples of 1Hz, whether you want them or not
2. Measurement results are scaled such that binning and amplitude of each result is even across frequency
3. The Loiacono Transform is defined in terms of real values. No Euler's relation, no complex numbers, no problem. 
4. The Loiacono Transform is defined in terms of normalized frequency f'. No radians, no problem. 
5. The Loiacono Transform has been simplified to a sum of dot products

With a single disadvantage:
1. You can't use the FFT. 

For many use cases, this is a non-issue, expecially when using Graphics Processing Unit (GPU) for Digital Signal Processing (DSP). 

The Loiacono Transform was developed for use in music analysis. In this configuration, the frequencies f' are chosen such that the frequency bins span 12-Tone Equal Temperment (TET) evenly. For example, each note (A, A#, B...) may be analyzed as 100 cents. This is opposed to the Discrete Fourier Transform (DFT), which is evenly spaced across frequency, thereby giving substantially lower resolution for notes of lower pitch. The Loiacono Transform is necessarily slower than the Fast Fourier Transform (FFT) (the standard implementation of the DFT), but this disadvantage is offset by modern hardware.

The Loiacono Transform finds primary application in music analysis. It was developed for use in a vocoder. 

## Examples

The Loiacono transform is good for distinct harmonic analysis

```
# load the wav file
y, sr = librosa.load("ahhD3.wav", sr=None)
# generate a Loiacono based on this SR
linst = Loiacono(
    sr=sr, midistart=30, midiend=128, subdivisionOfSemitone=2.0, multiple=80
)
# get a section in the middle of sample for processing
y = y[int(len(y) / 2) : int(len(y) / 2 + linst.DTFTLEN)]
linst.run(y)
print(linst.selectedNote)
linst.plot()
```

The above code yields

![image](https://user-images.githubusercontent.com/8158655/203856756-1ff0fe31-5c17-4ce4-b0fb-da9e55a5ffc2.png)

And returns the correctly identified note number, 50, or D3

By running "python whiteNoiseTest", you can see that the resultant spectrum is flat against white noise (after a square root adjustment)

![image](https://user-images.githubusercontent.com/8158655/203856338-a5f5fa8e-e37f-428f-8947-d8e420fcf18e.png)


## Derivation

Where the DTFT is defined as
$$X(ω) = \sum_{n=-∞}^∞ x[n]e^{-iωn} $$

The DFT applies the DTFT over a bounded time, provides the result in integer multiples k of a base frequency, and keeps a consistant length N across frequency calculations
$$X(k) = \sum_{n=0}^{N-1} x[n]e^{\frac{-i2\pi}{N}kn} $$

As an alternative to the DFT, the Loiacono transform selects frequencies that are evenly spaced in log space. This is of greater utility in music processing. Futhermore, the computation length for each frequency is an integer multiple m of the corresponding period. This causes even binning and response time across the frequency spectrum. Units of normalized frequency (or "cycles per sample") f' are used, because they are found to be much easier and more intuitive in implementation. Period p' ("samples per cycle") is given as 1/f'. 
$$L(f') = \sum_{n=0}^{p'm} x[n]e^{-i2\pi f' n} $$

Applying Euler's formula, this can be expressed in terms of real values
$$L(f') = \sum_{n=0}^{p'm} x[n][\cos({2\pi f' n}) - i\sin ({2\pi f' n})] $$

We can precompute the latter part of the expression as "twiddle factors" T, where 

$$
\begin{equation}
T(f',n) = 
    \begin{array}{lr}
        cos({2\pi f' n}) - i\sin ({2\pi f' n}), & \text{if } n < p'm \\
        0, & otherwise
    \end{array}
\end{equation}
$$

Zeros are introduced to the array to keep all arrays of equal length, which makes computation much easier for the programmer. 

Now the Loiacono transform can be expressed more simply as 
$$L(f') = \sum_{n=0}^{p'm} x[n]T(f',n)$$

For convenience, we separate the real and imaginary values of T

$$
\begin{equation}
T_r(f',n) = 
    \begin{array}{lr}
        cos({2\pi f' n}), & \text{if } n < p'm \\
        0, & otherwise
    \end{array}
\end{equation}
$$

$$
\begin{equation}
T_i(f',n) = 
    \begin{array}{lr}
        -\sin ({2\pi f' n}), & \text{if } n < p'm \\
        0, & otherwise
    \end{array}
\end{equation}
$$

Now the two axes of the Loiacono transform can be specified as

$$L_r(f') = \sum_{n=0}^{p'm} x[n]T_r(f',n)$$

$$L_i(f') = \sum_{n=0}^{p'm} x[n]T_i(f',n)$$

Or, as a simple dot product 

$$L_r(f') = x \cdot T_r $$

$$L_i(f') = x \cdot T_i $$


Therefore, the target phase and magnitude of each frequency are

$$ angle(L_r, L_i) $$ 

and 

$$ |L_r+L_i| $$

For musical considerations, we define the range of f' to be 

f' = [A*2**((NOTE-69)/12)/SAMPLE_FREQUENCY for NOTE in MIDIINDICES]

where, for example A = 440 Hz, SAMPLE_FREQUENCY = 48 kHz
MIDIINDICES may be 
1,2,3... 128
or may be more fine, ex. 
0.5, 1.0, 1.5, 2.0 ... 128

You can select any set you want! The above is even in log-space. 
