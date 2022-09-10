here = os.path.dirname(os.path.abspath(__file__))

localtest = True
if localtest == True:
    vkpath = os.path.join(here, "..", "vulkanese")
    # sys.path.append(vkpath)
    sys.path = [vkpath] + sys.path
    print(vkpath)
    print(sys.path)
    from vulkanese import Instance
    from vulkanese import *
else:
    from vulkanese.vulkanese import *
    
    
class SpectralShader:
    def __init__(self):

        self.pcmBufferOut = Buffer(
            binding=0,
            device=device,
            type="float",
            descriptorSet=device.descriptorPool.descSetGlobal,
            qualifier="in",
            name="pcmBufferOut",
            readFromCPU=True,
            SIZEBYTES=4 * 4 * self.SAMPLES_PER_DISPATCH * self.CHANNELS,
            usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            location=0,
            format=VK_FORMAT_R32_SFLOAT,
        )

        self.noteBasePhase = Buffer(
            binding=1,
            device=device,
            type="float",
            descriptorSet=device.descriptorPool.descSetUniform,
            qualifier="",
            name="noteBasePhase",
            readFromCPU=True,
            SIZEBYTES=4 * 4 * self.POLYPHONY,
            usage=VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            location=0,
            format=VK_FORMAT_R32_SFLOAT,
        )

        self.noteBaseIncrement = Buffer(
            binding=2,
            device=device,
            type="float",
            descriptorSet=device.descriptorPool.descSetUniform,
            qualifier="",
            name="noteBaseIncrement",
            readFromCPU=True,
            SIZEBYTES=4 * 4 * self.POLYPHONY,
            usage=VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            location=0,
            format=VK_FORMAT_R32_SFLOAT,
        )

        self.partialMultiplier = Buffer(
            binding=3,
            device=device,
            type="float",
            descriptorSet=device.descriptorPool.descSetUniform,
            qualifier="",
            name="partialMultiplier",
            readFromCPU=True,
            SIZEBYTES=4 * 4 * self.PARTIALS_PER_VOICE,
            usage=VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            location=0,
            format=VK_FORMAT_R32_SFLOAT,
        )

        self.partialVolume = Buffer(
            binding=4,
            device=device,
            type="float",
            descriptorSet=device.descriptorPool.descSetUniform,
            qualifier="",
            readFromCPU=True,
            name="partialVolume",
            SIZEBYTES=4 * 4 * self.PARTIALS_PER_VOICE,
            usage=VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            location=0,
            format=VK_FORMAT_R32_SFLOAT,
        )

        self.noteVolume = Buffer(
            binding=5,
            device=device,
            type="float",
            descriptorSet=device.descriptorPool.descSetUniform,
            qualifier="",
            name="noteVolume",
            readFromCPU=True,
            SIZEBYTES=4 * 4 * self.POLYPHONY,
            usage=VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            location=0,
            format=VK_FORMAT_R32_SFLOAT,
        )

        self.noteStrikeTime = Buffer(
            binding=6,
            device=device,
            type="float64_t",
            descriptorSet=device.descriptorPool.descSetUniform,
            qualifier="",
            name="noteStrikeTime",
            readFromCPU=True,
            SIZEBYTES=4 * 8 * self.POLYPHONY,
            usage=VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            location=0,
            format=VK_FORMAT_R32_SFLOAT,
        )

        self.noteReleaseTime = Buffer(
            binding=7,
            device=device,
            type="float64_t",
            descriptorSet=device.descriptorPool.descSetUniform,
            qualifier="",
            name="noteReleaseTime",
            readFromCPU=True,
            SIZEBYTES=4 * 8 * self.POLYPHONY,
            usage=VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            location=0,
            format=VK_FORMAT_R32_SFLOAT,
        )

        self.currTime = Buffer(
            binding=8,
            device=device,
            type="float64_t",
            descriptorSet=device.descriptorPool.descSetUniform,
            qualifier="",
            name="currTime",
            readFromCPU=True,
            SIZEBYTES=4 * 4 * self.POLYPHONY,
            usage=VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            location=0,
            format=VK_FORMAT_R32_SFLOAT,
        )

        self.attackEnvelope = Buffer(
            binding=9,
            device=device,
            type="float",
            descriptorSet=device.descriptorPool.descSetUniform,
            qualifier="",
            name="attackEnvelope",
            readFromCPU=True,
            SIZEBYTES=4 * 4 * self.ENVELOPE_LENGTH * self.POLYPHONY,
            usage=VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            location=0,
            format=VK_FORMAT_R32_SFLOAT,
        )

        self.releaseEnvelope = Buffer(
            binding=10,
            device=device,
            type="float",
            descriptorSet=device.descriptorPool.descSetUniform,
            qualifier="",
            name="releaseEnvelope",
            readFromCPU=True,
            SIZEBYTES=4 * 4 * self.ENVELOPE_LENGTH * self.POLYPHONY,
            usage=VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            location=0,
            format=VK_FORMAT_R32_SFLOAT,
        )
        for v in range(self.POLYPHONY):
            # minimum read 16 bytes,  * ENVELOPE_LENGTH

            VOICEBASE = v * 16 * self.ENVELOPE_LENGTH

            # "ATTACK_TIME" : 0, ALL TIME AS A FLOAT OF SECONDS
            self.attackEnvelope.pmap[
                VOICEBASE + 0 * 4 : VOICEBASE + 0 * 4 + 4
            ] = np.array([0.25], dtype=np.float32)
            # "ATTACK_LEVEL" : 1,
            self.attackEnvelope.pmap[
                VOICEBASE + 1 * 4 : VOICEBASE + 1 * 4 + 4
            ] = np.array([1.0], dtype=np.float32)
            # "DECAY_TIME"  : 2,
            self.attackEnvelope.pmap[
                VOICEBASE + 2 * 4 : VOICEBASE + 2 * 4 + 4
            ] = np.array([0.5], dtype=np.float32)
            # "DECAY_LEVEL"  : 3,
            self.attackEnvelope.pmap[
                VOICEBASE + 3 * 4 : VOICEBASE + 3 * 4 + 4
            ] = np.array([0.75], dtype=np.float32)
            # "RELEASE_TIME": 4,
            self.attackEnvelope.pmap[
                VOICEBASE + 4 * 4 : VOICEBASE + 4 * 4 + 4
            ] = np.array([4.0], dtype=np.float32)
            # "RELEASE_LEVEL": 5,
            self.attackEnvelope.pmap[
                VOICEBASE + 5 * 4 : VOICEBASE + 5 * 4 + 4
            ] = np.array([0], dtype=np.float32)

        self.freqFilter = Buffer(
            binding=10,
            device=device,
            type="float",
            descriptorSet=device.descriptorPool.descSetUniform,
            qualifier="",
            name="freqFilter",
            readFromCPU=True,
            SIZEBYTES=4 * 4 * self.FILTER_STEPS,
            usage=VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            location=0,
            format=VK_FORMAT_R32_SFLOAT,
        )
        self.freqFilter.pmap[:] = np.ones(4*self.FILTER_STEPS, dtype=np.float32)

        self.pitchFactor = Buffer(
            binding=11,
            device=device,
            type="float",
            descriptorSet=device.descriptorPool.descSetUniform,
            qualifier="",
            name="pitchFactor",
            readFromCPU=True,
            SIZEBYTES=4 * 4 * self.POLYPHONY,
            usage=VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            location=0,
            format=VK_FORMAT_R32_SFLOAT,
        )

        if self.GRAPH:
            # to run GUI event loop
            plt.ion()

            # here we are creating sub plots
            self.figure, ax = plt.subplots(figsize=(10, 8))
            self.plot, = ax.plot(self.freqFilterTotal[0 : 4 * self.FILTER_STEPS])
            plt.ylabel("some numbers")
            plt.show()
            plt.ylim(-2, 2)

        header = """#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float64 : enable
//#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_ARB_separate_shader_objects : enable
        """
        for k, v in self.replaceDict.items():
            header += "#define " + k + " " + str(v) + "\n"
        header += "layout (local_size_x = 1, local_size_y = PARTIALS_PER_VOICE, local_size_z = 1 ) in;"

        main = """
        void main() {

          uint timeSlice = gl_GlobalInvocationID.y;

          float sum = 0;
          
          for (uint noteNo = 0; noteNo<POLYPHONY; noteNo++){
          
              // calculate the envelope 
              // time is a float holding seconds (since epoch?)
              // these values are updated in the python loop
              float env = 0;
              // attack phase
              float64_t secondsSinceStrike  = abs(currTime[0] - noteStrikeTime[noteNo] );
              float64_t secondsSinceRelease = abs(currTime[0] - noteReleaseTime[noteNo]);
              if(noteStrikeTime[noteNo] > noteReleaseTime[noteNo]){
              
                // if envelope is complete, maintain at the final index
                // simply follow the path in memory
                
              }
              // release phase
              else{
                //ENVELOPE_LENGTH
              }
              
              // the note volume is given, and env is applied as well
              float noteVol = noteVolume[noteNo] * 1;
              
              
              float increment = noteBaseIncrement[noteNo]*pitchFactor[0];
              float phase = noteBasePhase[noteNo] + (timeSlice * increment);

              float innersum = 0;
              for (uint partialNo = 0; partialNo<PARTIALS_PER_VOICE; partialNo++)
              {
                float vol = partialVolume[partialNo];

                float harmonicRatio   = partialMultiplier[partialNo];
                float thisIncrement = increment * harmonicRatio;
                
                if(thisIncrement < 3.14){
                    int indexInFilter = int(thisIncrement*(FILTER_STEPS/(3.14)));
                    innersum += vol * sin(phase*harmonicRatio) * freqFilter[indexInFilter];
                }

              }
              sum+=innersum*noteVol;
          }
          
          pcmBufferOut[timeSlice] = sum/64;//(PARTIALS_PER_VOICE*POLYPHONY);
        }
        """

        # Stage
        existingBuffers = []
        mandleStage = Stage(
            device=device,
            name="mandlebrot.comp",
            stage=VK_SHADER_STAGE_COMPUTE_BIT,
            existingBuffers=existingBuffers,
            outputWidthPixels=700,
            outputHeightPixels=700,
            header=header,
            main=main,
            buffers=[
                self.pcmBufferOut,
                self.noteBasePhase,
                self.noteBaseIncrement,
                self.partialMultiplier,
                self.partialVolume,
                self.noteVolume,
                self.noteStrikeTime,
                self.noteReleaseTime,
                self.currTime,
                self.attackEnvelope,
                self.freqFilter,
                self.pitchFactor,
            ],
        )
