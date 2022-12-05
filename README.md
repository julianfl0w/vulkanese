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

For completeness, here is the associated GLSL template
```c
// From https://github.com/linebender/piet-gpu/blob/prefix/piet-gpu-hal/examples/shader/prefix.comp
// See https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf

#version 450
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_memory_scope_semantics : enable
// #extension VK_EXT_shader_atomic_float : require NOT WORKING

#define PI 3.1415926

DEFINE_STRING// This will be (or has been) replaced by constant definitions
BUFFERS_STRING// This will be (or has been) replaced by buffer definitions
    
layout (local_size_x = THREADS_PER_WORKGROUP, local_size_y = 1, local_size_z = 1 ) in;


void main(){
    
    // subgroupSize is the size of the subgroup – matches the API property
    //gl_SubgroupInvocationID is the ID of the invocation within the subgroup, an integer in the range [0..gl_SubgroupSize).
    // gl_SubgroupID is the ID of the subgroup within the local workgroup, an integer in the range [0..gl_NumSubgroups).
    //gl_NumSubgroups is the number of subgroups within the local workgroup.

    uint workGroup_ix       = gl_WorkGroupID.x;
    uint thread_ix          = gl_LocalInvocationID.x;
    uint workgroupStart_ix  = workGroup_ix*THREADS_PER_WORKGROUP;
    uint absoluteSubgroupId = gl_SubgroupID + gl_NumSubgroups * workGroup_ix;
    uint unique_thread_ix   = absoluteSubgroupId*gl_SubgroupSize + gl_SubgroupInvocationID;
    uint n                  = unique_thread_ix%SIGNAL_LENGTH;
    uint read_ix            = (unique_thread_ix+offset[0])%SIGNAL_LENGTH;
    uint frequency_ix       = unique_thread_ix/SIGNAL_LENGTH;
    
    float Tr = 0;
    float Ti = 0;
    
    float thisF     = f[frequency_ix];
    float thisP     = 1/thisF;
    if(n >= SIGNAL_LENGTH - multiple*thisP){
        
        // do the loiacono transform
        float thisDatum = x[read_ix];
        float dftlen = 1/sqrt(multiple / thisF);
        
        Tr =  thisDatum*cos(2*PI*thisF*n)*dftlen;
        Ti = -thisDatum*sin(2*PI*thisF*n)*dftlen;
    }
    
    
    // first reduction
    float TrSum = subgroupAdd(Tr);
    float TiSum = subgroupAdd(Ti);
    
    if (subgroupElect()) {
        Lr1[absoluteSubgroupId] = TrSum;
        Li1[absoluteSubgroupId] = TiSum;
    }
    
    // second stage reduction
    if(absoluteSubgroupId >= n*gl_SubgroupSize){
        return;
    }
    
    barrier();
    memoryBarrierBuffer();
    
    TrSum = subgroupAdd(Lr1[unique_thread_ix]);
    TiSum = subgroupAdd(Li1[unique_thread_ix]);
    if (subgroupElect()) {
        Lr0[absoluteSubgroupId] = TrSum;
        Li0[absoluteSubgroupId] = TiSum;
    }

    // third stage reduction
    if(absoluteSubgroupId >= n){
        return;
    }
    
    barrier();
    memoryBarrierBuffer();
    
    TrSum = subgroupAdd(Lr0[unique_thread_ix]);
    TiSum = subgroupAdd(Li0[unique_thread_ix]);
    
    if (subgroupElect()) {
        L[absoluteSubgroupId] = sqrt(TrSum*TrSum + TiSum*TiSum);
        //L[absoluteSubgroupId] = 1;
    }
}
```
  
