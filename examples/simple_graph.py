# include the Vulkanese directory
# (in case you're doing local development)
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "sinode"))
)
import vulkanese as ve
import numpy as np
import sinode.sinode as sinode

"""the compute graph representing
 result=(v+w)*(x+y)
"""
class SimpleGraph(sinode.Sinode):
    def __init__(self, **kwargs):
        sinode.Sinode.__init__(self, **kwargs)

        # first, declare all input buffers
        self.v = self.device.getStorageBuffer(name="v", shape=[128])
        self.w = self.device.getStorageBuffer(name="w", shape=[128])
        self.x = self.device.getStorageBuffer(name="x", shape=[128])
        self.y = self.device.getStorageBuffer(name="y", shape=[128])

        # then declare all shaders
        # shaders create their own output buffers
        # (typically called "result")
        addShader0 = ve.math.arith.add(name="add0", x=self.v, y=self.w, device = self.device)
        addShader1 = ve.math.arith.add(name="add1", x=self.x, y=self.y, device = self.device)
        multiplyShader = ve.math.arith.multiply(
            name="multiply",
            x=addShader0.gpuBuffers.result,
            y=addShader1.gpuBuffers.result,
            device=device,
            depends = [addShader0, addShader1]
        )
        self.result = multiplyShader.gpuBuffers.result
        self.shaders = [addShader0, addShader1, multiplyShader]
        for shader in self.shaders:
            shader.finalize()

    def run(self):
        for shader in self.shaders:
            shader.run()
        
# begin GPU test
instance = ve.instance.Instance(verbose=True)
device = instance.getDevice(0)
simpleGraph = SimpleGraph(device = device)
# set the inputs
simpleGraph.v.set(np.arange(128))
simpleGraph.w.set(np.arange(128))
simpleGraph.x.set(np.arange(128))
simpleGraph.y.set(np.arange(128))
simpleGraph.run()

print(simpleGraph.result.get())

instance.release()

