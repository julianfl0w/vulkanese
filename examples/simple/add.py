import os
import sys
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))

localtest = True
if localtest == True:
    vkpath = os.path.join(here, "..", "..", "vulkanese")
    sys.path = [vkpath] + sys.path
    from vulkanese import *
else:
    from vulkanese.vulkanese import *

# get vulkanese instance
instance_inst = Instance()

# any parameters set here will be available as
# param within this Python class
# and as a defined value within the shader
constantsDict = {"NUMBERS_TO_SUM": 2 ** 26, "SHADER_COUNT": 512}
# derived values
constantsDict["NUMBERS_PER_SHADER"] = int(
    constantsDict["NUMBERS_TO_SUM"] / constantsDict["SHADER_COUNT"]
)

# make constants available in this context
for k, v in constantsDict.items():
    exec(k + " = " + str(v))

# the output of the compute shader,
# which in our case is always a Storage Buffer
shaderOutputBuffers = [
    {"name": "sumOut", "type": "float64_t", "dims": ["SHADER_COUNT"]},
    # {"name": "bufferToSum", "type": "float64_t", "dims": ["NUMBERS_TO_SUM"]}
]

# choose a device
print("naively choosing device 0")
device = instance_inst.getDevice(0)

# heres how DIMENSIONS RELATE TO THEIR INDEX
dim2index = {"SHADER_COUNT": "shaderx", "NUMBERS_TO_SUM": "i"}

glslCode = """#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require
#extension GL_ARB_separate_shader_objects : enable
DEFINE_STRING  // This will be (or has been) replaced by constant definitions
BUFFERS_STRING // This will be (or has been) replaced by buffer definitions
layout (local_size_x = SHADER_COUNT, local_size_y = 1, local_size_z = 1 ) in;

void main() {
    uint shaderx = gl_LocalInvocationID.x;
    uint shadery = gl_LocalInvocationID.y;
    uint shaderz = gl_LocalInvocationID.z;
    VARIABLEDECLARATIONS
    float64_t shaderSum = 0;
    // simply loop over the numbers and add
    uint baseIndex = shaderx*NUMBERS_PER_SHADER;
    for(uint i = baseIndex; i < baseIndex+NUMBERS_PER_SHADER; i++){
        shaderSum += float64_t(i);
    }
    // only write to the storage buffer once
    sumOut[shaderx] = shaderSum;
}
"""
RECOMPILE = True
shader_basename = os.path.join("shaders", "add")
if RECOMPILE:
    sourceFilename = shader_basename + ".c"
else:
    sourceFilename = shader_basename + ".spv"

# Compute Stage: the only stage
computeShader = Stage(
    shaderCode = glslCode, # can be GLSL or SPIRV
    parent=instance_inst,
    constantsDict=constantsDict,
    device=device,
    name=shader_basename,
    stage=VK_SHADER_STAGE_COMPUTE_BIT,
    shaderOutputBuffers=shaderOutputBuffers,
    debuggableVars=[],
    shaderInputBuffers=[],
    shaderInputBuffersNoDebug=[],
    DEBUG=False,
    dim2index=dim2index,
)

# generate a compute cmd buffer
addPipeline = ComputePipeline(
    computeShader=computeShader,
    device=device,
    constantsDict=constantsDict,
)
        
# initialize numbers to sum
# (must be double size because minimum read in shader is 16 bytes)
# addPipeline.bufferToSum.setBuffer(np.arange(NUMBERS_TO_SUM * 2) / 2)
shaderStart = time.time()   
addPipeline.run()
shaderSum = np.sum(computeShader.sumOut.getAsNumpyArray())
shaderTime = time.time() - shaderStart

# initialize numpy version
numpersToSum = np.arange(NUMBERS_TO_SUM)
numpyStart = time.time()
npSum = sum(numpersToSum)
numpyTime = time.time() - numpyStart

print("Shader result " + str(shaderSum))
print("NumPy  result " + str(npSum))
print("Shader time " + str(shaderTime))
print("NumPy  time " + str(numpyTime))
print("Vulkanese speedup: " + str(numpyTime / shaderTime) + "x")

instance_inst.release()
#
