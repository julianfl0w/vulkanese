import os
import sys
import time
import numpy as np

arith_home = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import vulkanese as ve
import vulkan as vk


class ARITH(ve.shader.ComputeShader):
    def __init__(
        self,
        constantsDict,
        device,
        X,
        Y,
        DEBUG=False,
        buffType="float64_t",
        shader_basename="shaders/arith",
        memProperties=(
            vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        ),
    ):

        constantsDict["PROCTYPE"] = buffType
        constantsDict["YLEN"] = np.prod(np.shape(Y))
        constantsDict["LG_WG_SIZE"] = 7  # corresponding to 128 threads, a good number
        constantsDict["THREADS_PER_WORKGROUP"] = 1 << constantsDict["LG_WG_SIZE"]

        # device selection and instantiation
        self.instance = device.instance
        self.device = device
        self.constantsDict = constantsDict

        buffers = [
            ve.buffer.StorageBuffer(
                device=self.device,
                name="x",
                memtype=buffType,
                qualifier="readonly",
                dimensionVals=np.shape(X),
                memProperties=memProperties,
            ),
            ve.buffer.StorageBuffer(
                device=self.device,
                name="y",
                memtype=buffType,
                qualifier="readonly",
                dimensionVals=np.shape(Y),
                memProperties=memProperties,
            ),
            ve.buffer.StorageBuffer(
                device=self.device,
                name="sumOut",
                memtype=buffType,
                qualifier="writeonly",
                dimensionVals=np.shape(X),
                memProperties=memProperties,
            ),
        ]

        # Compute Stage: the only stage
        ve.shader.ComputeShader.__init__(
            self,
            sourceFilename=os.path.join(
                arith_home, shader_basename + ".c"
            ),  # can be GLSL or SPIRV
            parent=self.instance,
            constantsDict=self.constantsDict,
            device=self.device,
            name=shader_basename,
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            buffers=buffers,
            DEBUG=DEBUG,
            workgroupCount=[
                int(np.prod(np.shape(X)) / (constantsDict["THREADS_PER_WORKGROUP"])),
                1,
                1,
            ],
        )


def numpyTest(X, Y):

    # get numpy time, for comparison
    print("--- RUNNING NUMPY TEST ---")
    for i in range(10):
        nstart = time.time()
        nval = np.add(X, Y)
        nlen = time.time() - nstart
        print("Time " + str(nlen) + " seconds")
    return nval


def floatTest(X, Y, device, expectation):

    print("--- RUNNING GPU TEST ---")
    s = ARITH(
        {"OPERATION": "+"},
        device=device,
        X=X,
        Y=Y,
        buffType="float",
        memProperties=0 | vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
        # | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        | vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    )

    s.gpuBuffers.x.set(X)
    s.gpuBuffers.y.set(Y)
    for i in range(10):
        vstart = time.time()
        s.run()
        vlen = time.time() - vstart
        print("Time " + str(vlen) + " seconds")
    vval = s.gpuBuffers.sumOut.getAsNumpyArray()
    print("--- Testing for accuracy ---")
    print(np.allclose(expectation, vval))
    s.release()


if __name__ == "__main__":

    signalLen = 2 ** 23
    X = np.random.random((signalLen))
    Y = np.random.random((signalLen))

    # begin GPU test
    instance = ve.instance.Instance(verbose=False)
    device = instance.getDevice(0)
    nval = numpyTest(X, Y)
    floatTest(X, Y, device, expectation=nval)

    instance.release()
