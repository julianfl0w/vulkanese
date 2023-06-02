import os
import sys
import time
from scipy.signal import get_window

sum_home = os.path.dirname(os.path.abspath(__file__))
# if vulkanese isn't installed, check for a development version parallel to Loiacono repo ;)
import pkg_resources

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(here, "..", "..")))
import vulkanese as ve
import vulkan as vk


# Create a compute shader
class Sum(ve.shader.Shader):
    def __init__(
        self,
        device,
        inBuffer,
        N,
        waitSemaphores=[],
        constantsDict={},
        DEBUG=False,
        buffType="float",
        interleave=0,
        memProperties=0
        | vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        | vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
    ):

        # the constants will be placed into the shader.comp file,
        # and also available in Python
        constantsDict["INPUT_LENGTH"] = N * (device.subgroupSize ** 2)
        constantsDict["PROCTYPE"] = buffType
        constantsDict["TOTAL_THREAD_COUNT"] = N * N
        constantsDict["LG_WG_SIZE"] = 7
        constantsDict["THREADS_PER_WORKGROUP"] = 1 << constantsDict["LG_WG_SIZE"]
        constantsDict["N"] = N
        constantsDict["INTERLEAVE"] = interleave

        self.constantsDict = constantsDict

        # make the constants available locally
        for k, v in self.constantsDict.items():
            if type(v) != str:
                exec("self." + k + " = " + str(v))
            else:
                exec("self." + k + ' = "' + v + '"')

        # device selection and instantiation
        self.instance = device.instance
        self.device = device
        inBuffer.qualifier = "readonly"
        inBuffer.name = "inBuf"
        # declare buffers. they will be in GPU memory, but visible from the host (!)
        buffers = [
            # x is the input signal
            inBuffer,
            # The following 4 are reduction buffers
            # Intermediate buffers for computing the sum
            ve.buffer.StorageBuffer(
                device=self.device,
                name="Lr0",
                memtype=buffType,
                dimensionVals=[N, self.device.subgroupSize],
            ),
            # L is the final output
            ve.buffer.StorageBuffer(
                device=self.device,
                name="sumOut",
                memtype=buffType,
                qualifier="writeonly",
                dimensionVals=[N],
                memProperties=memProperties,
            ),
            # DebugBuffer(
            #    device=self.device,
            #    name="allShaders",
            #    memtype=buffType,
            #    dimensionVals=[constantsDict["TOTAL_THREAD_COUNT"]],
            # ),
        ]

        # Create a compute shader
        # Compute Stage: the only stage
        ve.shader.Shader.__init__(
            self,
            sourceFilename=os.path.join(
                sum_home, "shaders/sum.c"
            ),  # can be GLSL or SPIRV
            constantsDict=self.constantsDict,
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            device=self.device,
            name="sum",
            buffers=buffers,
            DEBUG=DEBUG,
            workgroupCount=[
                int(self.INPUT_LENGTH / (constantsDict["THREADS_PER_WORKGROUP"])),
                1,
                1,
            ],
            waitSemaphores=waitSemaphores,
            compressBuffers=True,  # flat float arrays, instead of skipping every 4
        )

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
        self.gpuBuffers.offset.setByIndex(index=0, data=[self.offset])
        self.run()
        self.absresult = self.gpuBuffers.L.getAsNumpyArray()
        return self.absresult


if __name__ == "__main__":

    # generate a sine wave at A440, SR=48000
    sr = 48000
    A4 = 440
    z = np.sin(np.arange(2 ** 15) * 2 * np.pi * A4 / sr)
    z += np.sin(2 * np.arange(2 ** 15) * 2 * np.pi * A4 / sr)
    z += np.sin(3 * np.arange(2 ** 15) * 2 * np.pi * A4 / sr)
    z += np.sin(4 * np.arange(2 ** 15) * 2 * np.pi * A4 / sr)

    multiple = 40
    normalizedStep = 5.0 / sr
    # create a linear distribution of desired frequencies
    fprime = np.arange(100 / sr, 3000 / sr, normalizedStep)

    # generate a Loiacono based on this SR
    # (this one runs in CPU. reference only)
    linst = Loiacono(fprime=fprime, multiple=multiple, dtftlen=2 ** 15)
    linst.debugRun(z)

    # begin GPU test
    instance = Instance(verbose=False)
    device = instance.getDevice(0)
    linst_gpu = Sum(device=device, fprime=fprime, multiple=linst.multiple)
    linst_gpu.gpuBuffers.x.set(z)
    for i in range(10):
        linst_gpu.debugRun()
    # linst_gpu.dumpMemory()
    readstart = time.time()
    linst_gpu.absresult = linst_gpu.gpuBuffers.L.getAsNumpyArray()
    print("Readtime " + str(time.time() - readstart))

    fig, ((ax1, ax2)) = plt.subplots(1, 2)
    ax1.plot(linst.fprime * sr, linst_gpu.absresult)
    ax1.set_title("GPU Result")
    ax2.plot(linst.fprime * sr, linst.absresult)
    ax2.set_title("CPU Result")

    plt.show()
