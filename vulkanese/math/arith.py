import os
import sys
import time
import numpy as np
import json

arith_home = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import vulkanese as ve
import vulkan as vk

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "sinode"))
)
import sinode.sinode as sinode


class ARITH(ve.shader.Shader):
    def __init__(self, **kwargs):
        sinode.Sinode.__init__(self, **kwargs)
        self.proc_kwargs(
            **{
                "parent": None,
                "DEBUG": False,
                "buffType": "float",
                "shader_basename": "shaders/arith",
                "memProperties": (
                    vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                    | vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                    | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                ),
                "useFence": True,
            }
        )

        constantsDict = {}
        constantsDict["PROCTYPE"] = self.buffType

        if hasattr(self, "OPERATION"):
            constantsDict["OPERATION"] = self.OPERATION
        elif hasattr(self, "FUNCTION1"):
            constantsDict["FUNCTION1"] = self.FUNCTION1
        elif hasattr(self, "FUNCTION2"):
            constantsDict["FUNCTION2"] = self.FUNCTION2
        else:
            die
        constantsDict["YLEN"] = np.prod(np.shape(self.Y))
        constantsDict["LG_WG_SIZE"] = 7  # corresponding to 128 threads, a good number
        constantsDict["THREADS_PER_WORKGROUP"] = 1 << constantsDict["LG_WG_SIZE"]

        # device selection and instantiation
        self.instance = self.device.instance
        self.constantsDict = constantsDict

        self.descriptorPool = ve.descriptor.DescriptorPool(
            device=self.device, parent=self
        )

        buffers = [
            ve.buffer.StorageBuffer(
                device=self.device,
                name="x",
                memtype=self.buffType,
                qualifier="readonly",
                dimensionVals=np.shape(self.X),
                memProperties=self.memProperties,
                descriptorSet=self.descriptorPool.descSetGlobal,
            ),
            ve.buffer.StorageBuffer(
                device=self.device,
                name="y",
                memtype=self.buffType,
                qualifier="readonly",
                dimensionVals=np.shape(self.Y),
                memProperties=self.memProperties,
                descriptorSet=self.descriptorPool.descSetGlobal,
            ),
            ve.buffer.StorageBuffer(
                device=self.device,
                name="sumOut",
                memtype=self.buffType,
                qualifier="writeonly",
                dimensionVals=np.shape(self.X),
                memProperties=self.memProperties,
                descriptorSet=self.descriptorPool.descSetGlobal,
            ),
        ]

        self.descriptorPool.finalize()

        # Compute Stage: the only stage
        ve.shader.Shader.__init__(
            self,
            sourceFilename=os.path.join(
                arith_home, self.shader_basename + ".c"
            ),  # can be GLSL or SPIRV
            constantsDict=self.constantsDict,
            device=self.device,
            name=self.shader_basename,
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            buffers=buffers,
            DEBUG=self.DEBUG,
            workgroupCount=[
                int(
                    np.prod(np.shape(self.X)) / (constantsDict["THREADS_PER_WORKGROUP"])
                ),
                1,
                1,
            ],
            useFence=self.useFence,
        )

        self.gpuBuffers.x.set(self.X)
        self.gpuBuffers.y.set(self.Y)

    def baseline(self, X, Y):
        if hasattr(self, "OPERATION"):
            retval = self.npEquivalent(X, Y)
            return retval
        elif hasattr(self, "FUNCTION1"):
            return eval("np." + self.FUNCTION1 + "(X)")
        elif hasattr(self, "FUNCTION2"):
            return eval("np." + self.FUNCTION2 + "(X, Y)")
        else:
            die

    def test(self):

        self.run(blocking=True)
        result = self.gpuBuffers.sumOut.get()
        expectation = self.baseline(self.gpuBuffers.x.get(), self.gpuBuffers.y.get())
        self.passed = np.allclose(result.astype(float), expectation.astype(float))

        if hasattr(self, "OPERATION"):
            print(self.OPERATION + ": " + str(self.passed))
        elif hasattr(self, "FUNCTION1"):
            print(self.FUNCTION1 + ": " + str(self.passed))
        elif hasattr(self, "FUNCTION2"):
            print(self.FUNCTION2 + ": " + str(self.passed))
        else:
            die

        return self.passed


def test(device):
    print("Testing Arithmatic")
    signalLen = 2 ** 10
    X = np.random.random((signalLen))
    Y = np.random.random((signalLen))
    toTest = [
        ARITH(
            parent=device, device=device, X=X, Y=Y, OPERATION="+", npEquivalent=np.add
        ),
        ARITH(device=device, X=X, Y=Y, OPERATION="-", npEquivalent=np.subtract),
        ARITH(device=device, X=X, Y=Y, OPERATION="*", npEquivalent=np.multiply),
        ARITH(device=device, X=X, Y=Y, OPERATION="/", npEquivalent=np.divide),
        ARITH(device=device, X=X, Y=Y, FUNCTION1="sin"),
        ARITH(device=device, X=X, Y=Y, FUNCTION1="cos"),
        ARITH(device=device, X=X, Y=Y, FUNCTION1="tan"),
        ARITH(device=device, X=X, Y=Y, FUNCTION1="exp"),
        # ARITH(device = device, X=X, Y=Y, FUNCTION1="asin"),
        # ARITH(device = device, X=X, Y=Y, FUNCTION1="acos"),
        # ARITH(device = device, X=X, Y=Y, FUNCTION1="atan"),
        ARITH(device=device, X=X, Y=Y, FUNCTION1="sqrt"),
        # ARITH(device = device, X=X, Y=Y, FUNCTION2="pow" ),
        # ARITH(device = device, X=X, Y=Y, FUNCTION2="mod" ),
        # ARITH(device = device, X=X, Y=Y, FUNCTION2="atan"),
    ]
    # print(json.dumps(device.asDict(), indent=2))

    for s in toTest:
        s.test()
        # s.release()


if __name__ == "__main__":

    # begin GPU test
    instance = ve.instance.Instance(verbose=True)
    device = instance.getDevice(0)

    test(device=device)
    instance.release()
