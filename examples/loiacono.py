import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import vulkanese as ve

# generate a sine wave at A440, SR=48000
# with 3 additional harmonics
siglen = 2 ** 15
sr = 48000
A4 = 440
z = np.sin(np.arange(siglen) * 2 * np.pi * A4 / sr)
z += np.sin(2 * np.arange(siglen) * 2 * np.pi * A4 / sr)
z += np.sin(3 * np.arange(siglen) * 2 * np.pi * A4 / sr)
z += np.sin(4 * np.arange(siglen) * 2 * np.pi * A4 / sr)

normalizedStep = 5.0 / sr
# create a linear distribution of desired frequencies
fprime = np.arange(100 / sr, 3000 / sr, normalizedStep)
# in constant-Q, select a multiple of the target period
multiple = 40

# get the Vulkan instance
instance = ve.instance.Instance(verbose=False)

# get the default device
device = instance.getDevice(0)

# create the shader manager
linst_gpu = ve.math.signals.loiacono.loiacono_gpu.Loiacono_GPU(
    device=device, fprime=fprime, multiple=multiple
)

# run the program
readstart = time.time()
linst_gpu.gpuBuffers.x.set(z)
linst_gpu.run(blocking=True)
print("Runtime " + str(time.time() - readstart))

linst_gpu.spectrum = linst_gpu.gpuBuffers.L.get()

# generate the plot
fig, ((ax1, ax2)) = plt.subplots(1, 2)
ax1.plot(np.arange(len(z)) / sr, z)
ax1.set_title("Signal")
ax2.plot(fprime, linst_gpu.spectrum)
ax2.set_title("Spectrum")
plt.show()
print(linst_gpu.spectrum)
# elegantly release everything
instance.release()
