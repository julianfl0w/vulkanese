# vulkanese  
## It's Vulkan-Ease!  

This repository  
* imposes a hierarchical structure on Vulkan
* dramatically simplifies Vulkan usage
* is pure python
* runs SPIR-V compute shaders efficiently, across all modern GPUs
* makes compute shader debugging easy

python helloTriangle.py   
![163123559-2410ed19-96be-495f-a172-f0221c8d9167](https://user-images.githubusercontent.com/8158655/163700475-7c18ba31-1e61-48d5-986c-08da9fec427d.png)
  
## The Hierarchy  
![vulkanese](https://user-images.githubusercontent.com/8158655/153063082-69028462-39de-4640-93ca-a3055b57a9ce.png)

## Installation  
1. python -m pip install git+https://github.com/julianfl0w/vulkan #Install the latest Vulkan Python wrapper
2. python -m pip install git+https://github.com/julianfl0w/vulkanese #Install this repo

## GPGPU Example  
I've implemented a world class pitch detector in GPU, based on the Loiacono Transform. That work can be found in the following repositories:
https://github.com/julianfl0w/loiacono
https://github.com/julianfl0w/gpuPitchDetect

Here is a snapshot of that code, which shows how to use Vulkanese to manage compute shaders:

```python
import os
import sys
import pkg_resources
import time

here = os.path.dirname(os.path.abspath(__file__))
# if vulkanese isn't installed, check for a development version parallel to Loiacono repo ;)
if "vulkanese" not in [pkg.key for pkg in pkg_resources.working_set]:
    sys.path = [os.path.join(here, "..", "vulkanese", "vulkanese")] + sys.path

from vulkanese import *
from loiacono import *

loiacono_home = os.path.dirname(os.path.abspath(__file__))

# Create a compute shader 
class Loiacono_GPU(ComputeShader):
    def __init__(
        self,
        device,
        fprime,
        multiple, 
        signalLength=2**15,
        constantsDict = {},
        DEBUG=False,
        buffType="float",
        memProperties=0
        | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
    ):

        # the constants will be placed into the shader.comp file, 
        # and also available in Python
        constantsDict["multiple"] = multiple
        constantsDict["SIGNAL_LENGTH"] = signalLength
        constantsDict["PROCTYPE"] = buffType
        constantsDict["TOTAL_THREAD_COUNT"] = signalLength * len(fprime)
        constantsDict["LG_WG_SIZE"] = 7
        constantsDict["THREADS_PER_WORKGROUP"] = 1 << constantsDict["LG_WG_SIZE"]
        self.dim2index = {}
        self.signalLength = signalLength

        # device selection and instantiation
        self.instance = device.instance
        self.device = device
        self.constantsDict = constantsDict
        self.numSubgroups = signalLength * len(fprime) / self.device.subgroupSize
        self.numSubgroupsPerFprime = int(self.numSubgroups / len(fprime))

        # declare buffers. they will be in GPU memory, but visible from the host (!)
        buffers = [
            # x is the input signal
            StorageBuffer(
                device=self.device,
                name="x",
                memtype=buffType,
                qualifier="readonly",
                dimensionVals=[2**15], # always 32**3
                memProperties=memProperties,
            ),
            # The following 4 are reduction buffers
            # Intermediate buffers for computing the sum 
            StorageBuffer(
                device=self.device,
                name="Li1",
                memtype=buffType,
                dimensionVals=[len(fprime), self.device.subgroupSize**2],
            ),
            StorageBuffer(
                device=self.device,
                name="Lr1",
                memtype=buffType,
                dimensionVals=[len(fprime), self.device.subgroupSize**2],
            ),
            StorageBuffer(
                device=self.device,
                name="Li0",
                memtype=buffType,
                dimensionVals=[len(fprime), self.device.subgroupSize],
            ),
            StorageBuffer(
                device=self.device,
                name="Lr0",
                memtype=buffType,
                dimensionVals=[len(fprime), self.device.subgroupSize],
            ),
            # L is the final output
            StorageBuffer(
                device=self.device,
                name="L",
                memtype=buffType,
                qualifier="writeonly",
                dimensionVals=[len(fprime)],
                memProperties=memProperties,
            ),
            StorageBuffer(
                device=self.device,
                name="f",
                memtype=buffType,
                qualifier="readonly",
                dimensionVals=[len(fprime)],
                memProperties=memProperties,
            ),
            StorageBuffer(
                device=self.device,
                name="offset",
                memtype="uint",
                qualifier="readonly",
                dimensionVals=[16],
                memProperties=memProperties,
            ),
            #DebugBuffer(
            #    device=self.device,
            #    name="allShaders",
            #    memtype=buffType,
            #    dimensionVals=[constantsDict["TOTAL_THREAD_COUNT"]],
            #),
        ]
        
        # Create a compute shader
        # Compute Stage: the only stage
        ComputeShader.__init__(
            self,
            sourceFilename=os.path.join(
                loiacono_home, "shaders/loiacono.c"
            ),  # can be GLSL or SPIRV
            parent=self.instance,
            constantsDict=self.constantsDict,
            device=self.device,
            name="loiacono",
            stage=VK_SHADER_STAGE_COMPUTE_BIT,
            buffers=buffers,
            DEBUG=DEBUG,
            dim2index=self.dim2index,
            workgroupCount=[
                int(
                    signalLength * len(fprime)
                    / (
                        constantsDict["THREADS_PER_WORKGROUP"]
                    )
                ),
                1,
                1,
            ],
            compressBuffers=True, # flat float arrays, instead of skipping every 4
        )
                
        self.gpuBuffers.f.set(fprime)
        self.gpuBuffers.offset.zeroInitialize()
        self.offset = 0

    def debugRun(self):
        vstart = time.time()
        self.run()
        vlen = time.time() - vstart
        self.absresult = self.gpuBuffers.L
        print("vlen " + str(vlen))
        # return self.sumOut.getAsNumpyArray()

    def feed(self, newData):
        self.gpuBuffers.x.setByIndexStart(self.offset, newData)
        self.offset = (self.offset + len(newData)) % self.signalLength
        self.gpuBuffers.offset.setByIndex(index = 0, data=[self.offset])
        self.run()
        return self.gpuBuffers.L.getAsNumpyArray()

if __name__ == "__main__":

    # generate a sine wave at A440, SR=48000
    sr = 48000
    A4 = 440
    z = np.sin(np.arange(2**15)*2*np.pi*A4/sr)
    
    
    multiple = 40
    normalizedStep = 1.0/sr
    # create a linear distribution of desired frequencies
    fprime = np.arange(100/sr,1000/sr,normalizedStep)
    
    # generate a Loiacono based on this SR
    # (this one runs in CPU. reference only)
    linst = Loiacono(
        fprime = fprime,
        multiple=multiple,
        dtftlen=2**15
    )
    linst.debugRun(z)
    
    # begin GPU test
    instance = Instance(verbose=False)
    device = instance.getDevice(0)
    linst_gpu = Loiacono_GPU(
        device = device,
        fprime = fprime,
        multiple = linst.multiple,
    )
    linst_gpu.gpuBuffers.x.set(z)
    for i in range(10):
        linst_gpu.debugRun()
    #linst_gpu.dumpMemory()
    readstart = time.time()
    linst_gpu.absresult = linst_gpu.gpuBuffers.L.getAsNumpyArray()
    print("Readtime " + str(time.time()- readstart))
    
    fig, ((ax1, ax2)) = plt.subplots(1, 2)
    ax1.plot(linst.fprime*sr, linst_gpu.absresult)
    ax1.set_title("GPU Result")
    ax2.plot(linst.fprime*sr, linst.absresult)
    ax2.set_title("CPU Result")
    
    plt.show()
    
```

We have sucessfully detected the 440Hz signal in this simple example:
![image](https://user-images.githubusercontent.com/8158655/205408263-2ab2236b-1b76-4f7d-9f6c-e4813ccb12d7.png)

  
