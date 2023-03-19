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
        sinode.Sinode.__init__(self, parent = kwargs["device"], **kwargs)
        self.proc_kwargs(
            **{
                "DEBUG": False,
                "buffers": [],
                "buffType": "float",
                "shader_basename": "shaders/arith",
                "memProperties": (
                    vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                    | vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                    | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                ),
            }
        )

        constantsDict = {}
        constantsDict["PROCTYPE"] = self.buffType

        if hasattr(self, "operation"):
            constantsDict["operation"] = self.operation
        elif hasattr(self, "FUNCTION1"):
            constantsDict["FUNCTION1"] = self.FUNCTION1
        elif hasattr(self, "FUNCTION2"):
            constantsDict["FUNCTION2"] = self.FUNCTION2
        else:
            die
        constantsDict["YLEN"] = np.prod(np.shape(self.y))
        constantsDict["LG_WG_SIZE"] = 7  # corresponding to 128 threads, a good number
        constantsDict["THREADS_PER_WORKGROUP"] = 1 << constantsDict["LG_WG_SIZE"]

        # device selection and instantiation
        self.instance = self.device.instance
        self.constantsDict = constantsDict

        if isinstance(self.x, ve.buffer.StorageBuffer):
            self.buffers = [self.x, self.y]
            self.x.name = "x"
            self.y.name = "y"
            
        else:
            self.buffers += [
                self.device.getStorageBuffer(
                    name="x",
                    memtype=self.buffType,
                    qualifier="readonly",
                    shape=np.shape(self.x),
                    memProperties=self.memProperties,
                ),
                self.device.getStorageBuffer(
                    name="y",
                    memtype=self.buffType,
                    qualifier="readonly",
                    shape=np.shape(self.y),
                    memProperties=self.memProperties,
                )]
            
            self.buffers[0].set(self.x)
            self.buffers[1].set(self.y)
        
        self.buffers += [
            self.device.getStorageBuffer(
                name="result",
                memtype=self.buffType,
                qualifier="writeonly",
                shape=self.buffers[0].shape,
                memProperties=self.memProperties,
            ),
        ]


        # Compute Stage: the only stage
        ve.shader.Shader.__init__(
            self,
            sourceFilename=os.path.join(
                arith_home, self.shader_basename + ".comp.template"
            ),  # can be GLSL or SPIRV
            constantsDict=self.constantsDict,
            device=self.device,
            name=self.shader_basename,
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            buffers=self.buffers,
            DEBUG=self.DEBUG,
            workgroupCount=[
                int(
                    np.prod(self.x.shape) / (constantsDict["THREADS_PER_WORKGROUP"])
                ),
                1,
                1,
            ],
        )


    def baseline(self, X, Y):
        if hasattr(self, "operation"):
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
        result = self.gpuBuffers.result.get()
        expectation = self.baseline(self.gpuBuffers.x.get(), self.gpuBuffers.y.get())
        self.passed = np.allclose(result.astype(float), expectation.astype(float))

        if hasattr(self, "operation"):
            print(self.operation + ": " + str(self.passed))
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
    x = np.random.random((signalLen))
    y = np.random.random((signalLen))
    toTest = [
        ARITH(
            device=device, x=x, y=y, operation="+", npEquivalent=np.add
        ),
        ARITH(device=device, x=x, y=y, operation="-", npEquivalent=np.subtract),
        ARITH(device=device, x=x, y=y, operation="*", npEquivalent=np.multiply),
        ARITH(device=device, x=x, y=y, operation="/", npEquivalent=np.divide),
        ARITH(device=device, x=x, y=y, FUNCTION1="sin"),
        ARITH(device=device, x=x, y=y, FUNCTION1="cos"),
        ARITH(device=device, x=x, y=y, FUNCTION1="tan"),
        ARITH(device=device, x=x, y=y, FUNCTION1="exp"),
        # ARITH(device = device, x=x, y=y, FUNCTION1="asin"),
        # ARITH(device = device, x=x, y=y, FUNCTION1="acos"),
        # ARITH(device = device, x=x, y=y, FUNCTION1="atan"),
        ARITH(device=device, x=x, y=y, FUNCTION1="sqrt"),
        # ARITH(device = device, x=x, y=y, FUNCTION2="pow" ),
        # ARITH(device = device, x=x, y=y, FUNCTION2="mod" ),
        # ARITH(device = device, x=x, y=y, FUNCTION2="atan"),
    ]
    # print(json.dumps(device.asDict(), indent=2))

    for s in toTest:
        s.finalize()
        s.test()
        # s.release()

class add(ARITH):
    def __init__(self, **kwargs):
        kwargs["operation"] = "+"
        ARITH.__init__(self, **kwargs)

class multiply(ARITH):
    def __init__(self, **kwargs):
        kwargs["operation"] = "*"
        ARITH.__init__(self, **kwargs)


if __name__ == "__main__":

    # begin GPU test
    instance = ve.instance.Instance(verbose=True)
    device = instance.getDevice(0)

    test(device=device)
    instance.release()
