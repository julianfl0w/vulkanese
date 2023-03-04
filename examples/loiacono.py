import sys
import os 
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import vulkanese as ve

# generate a sine wave at A440, SR=48000
# with 3 additional harmonics
sr = 48000
A4 = 440
z = np.sin(np.arange(2**15)*2*np.pi*A4/sr)
z += np.sin(2*np.arange(2**15)*2*np.pi*A4/sr)
z += np.sin(3*np.arange(2**15)*2*np.pi*A4/sr)
z += np.sin(4*np.arange(2**15)*2*np.pi*A4/sr)

normalizedStep = 5.0/sr
# create a linear distribution of desired frequencies
fprime = np.arange(100/sr,3000/sr,normalizedStep)
# in constant-Q, select a multiple of the target period
multiple = 40

# get the Vulkan instance
instance = ve.instance.Instance(verbose=False)

# get the default device
device = instance.getDevice(0)

# create the shader manager
linst_gpu = ve.math.loiacono.loiacono_gpu.Loiacono_GPU(
    device = device,
    fprime = fprime,
    multiple = multiple,
)

readstart = time.time()
linst_gpu.run(z)
print("Runtime " + str(time.time()- readstart))
    
linst_gpu.spectrum = linst_gpu.gpuBuffers.L.getAsNumpyArray()

# generate the plot
fig, ((ax1, ax2)) = plt.subplots(1, 2)
ax1.plot(linst.fprime*sr, linst_gpu.spectrum)
ax1.set_title("Spectrum")
plt.show()

instance.release()
