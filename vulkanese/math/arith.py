import os
import sys
import time
import numpy as np

arith_home = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import vulkanese as ve
import vulkan as vk


class ARITH(ve.shader.Shader):
    def __init__(
        self,
        device,
        X,
        Y,
        OPERATION=None,
        FUNCTION1=None,
        FUNCTION2=None,
        npEquivalent=None,
        DEBUG=False,
        buffType="float",
        shader_basename="shaders/arith",
        memProperties=(
            vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        ),
        useFence=True, 
    ):
        constantsDict = {}
        constantsDict["PROCTYPE"] = buffType
        if OPERATION is not None:
            constantsDict["OPERATION"] = OPERATION
        if FUNCTION1 is not None:
            constantsDict["FUNCTION1"] = FUNCTION1
        if FUNCTION2 is not None:
            constantsDict["FUNCTION2"] = FUNCTION2
        constantsDict["YLEN"] = np.prod(np.shape(Y))
        constantsDict["LG_WG_SIZE"] = 7  # corresponding to 128 threads, a good number
        constantsDict["THREADS_PER_WORKGROUP"] = 1 << constantsDict["LG_WG_SIZE"]

        # device selection and instantiation
        self.npEquivalent = npEquivalent
        self.OPERATION = OPERATION
        self.FUNCTION1 = FUNCTION1
        self.FUNCTION2 = FUNCTION2
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

        self.device.descriptorPool.finalize()

        # Compute Stage: the only stage
        ve.shader.Shader.__init__(
            self,
            sourceFilename=os.path.join(
                arith_home, shader_basename + ".c"
            ),  # can be GLSL or SPIRV
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
            useFence = useFence
        )

        self.gpuBuffers.x.set(X)
        self.gpuBuffers.y.set(Y)
        
    def baseline(self, X, Y):
        if self.OPERATION is not None:
            retval = self.npEquivalent(X,Y)
            return retval
        if self.FUNCTION1 is not None:
            return eval("np." + self.FUNCTION1  + "(X)")
        if self.FUNCTION2 is not None:
            return eval("np." + self.FUNCTION2  + "(X, Y)")
            

    def test(self):

        self.run(blocking=True)
        result = self.gpuBuffers.sumOut.get()
        expectation = self.baseline(
            self.gpuBuffers.x.get(),
            self.gpuBuffers.y.get()
            )
        self.passed = np.allclose(result.astype(float), expectation.astype(float))
        if self.OPERATION is not None:
            print(self.OPERATION + ": " + str(self.passed))
        if self.FUNCTION1 is not None:
            print(self.FUNCTION1 + ": " + str(self.passed))
        if self.FUNCTION2 is not None:
            print(self.FUNCTION2 + ": " + str(self.passed))
        return self.passed


def test(device):
    print("Testing Arithmatic")
    signalLen = 2 ** 12
    X = np.random.random((signalLen))
    Y = np.random.random((signalLen))
    toTest = [
        ARITH(device = device, X=X, Y=Y, OPERATION="+"   , npEquivalent=np.add),
        ARITH(device = device, X=X, Y=Y, OPERATION="-"   , npEquivalent=np.subtract),
        ARITH(device = device, X=X, Y=Y, OPERATION="*"   , npEquivalent=np.multiply),
        ARITH(device = device, X=X, Y=Y, OPERATION="/"   , npEquivalent=np.divide),
        ARITH(device = device, X=X, Y=Y, FUNCTION1="cos" ),
        ARITH(device = device, X=X, Y=Y, FUNCTION1="sin" ),
        ARITH(device = device, X=X, Y=Y, FUNCTION1="tan" ),
        ARITH(device = device, X=X, Y=Y, FUNCTION1="exp" ),
        #ARITH(device = device, X=X, Y=Y, FUNCTION1="asin"),
        #ARITH(device = device, X=X, Y=Y, FUNCTION1="acos"),
        #ARITH(device = device, X=X, Y=Y, FUNCTION1="atan"),
        ARITH(device = device, X=X, Y=Y, FUNCTION1="sqrt"),
        #ARITH(device = device, X=X, Y=Y, FUNCTION2="pow" ),
        ARITH(device = device, X=X, Y=Y, FUNCTION2="mod" ),
        #ARITH(device = device, X=X, Y=Y, FUNCTION2="atan"),
    ]
    for s in toTest:
        s.test()
        #s.release()


if __name__ == "__main__":

    # begin GPU test
    instance = ve.instance.Instance(verbose=False)
    device = instance.getDevice(0)
    
    test(device=device)
    instance.release()
