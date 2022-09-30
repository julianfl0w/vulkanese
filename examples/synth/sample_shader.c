#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require
//#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_ARB_separate_shader_objects : enable
DEFINE_STRING // This will be (or has been) replaced by constant definitions
layout (local_size_x = SAMPLES_PER_DISPATCH, local_size_y = SHADERS_PER_SAMPLE, local_size_z = 1 ) in;
BUFFERS_STRING  // This will be (or has been) replaced by buffer definitions
void main() {
  //precision mediump float;
  //precision mediump float64_t;
  uint sampleNo = gl_LocalInvocationID.x;
  uint shaderIndexInSample = gl_LocalInvocationID.y;
  uint zindex = gl_LocalInvocationID.z;
  VARIABLEDECLARATIONS

  // current time depends on the sample offset
  float64_t currOffset = sampleNo / float(SAMPLE_FREQUENCY);
  currTimeWithSampleOffset = currTime[0] + currOffset;
  // currTimeWithSampleOffset = shaderIndexInSample;
  // //sampleNo*SHADERS_PER_SAMPLE+shaderIndexInSample;

  for (uint noteNo = shaderIndexInSample * POLYPHONY_PER_SHADER;
       noteNo < (shaderIndexInSample + 1) * POLYPHONY_PER_SHADER; noteNo++) {

    // calculate the envelope
    // time is a float holding seconds (since epoch?)
    // these values are updated in the python loop

    // attack phase
    secondsSinceStrike  = currTimeWithSampleOffset - noteStrikeTime[noteNo];
    secondsSinceRelease = currTimeWithSampleOffset - noteReleaseTime[noteNo];

    // attack phase
    if (noteStrikeTime[noteNo] > noteReleaseTime[noteNo]) {
      envelopeIndexFloat64 = secondsSinceStrike * attackSpeedMultiplier[noteNo];
    }
    // release phase
    else if (noteStrikeTime[noteNo] < noteReleaseTime[noteNo]) {
      envelopeIndexFloat64 =
          secondsSinceRelease * releaseSpeedMultiplier[noteNo];
    }
    // if both strike- and release-time are 0
    // continue to next one
    else {
      continue;
    }
    
    // keep the envelopeIndexFractional part, for interp
    envelopeIndexFractional = fract(envelopeIndexFloat64);
    envelopeIndex = int(envelopeIndexFloat64);

    // if envelope is complete, maintain at the second-to-final index
    if (envelopeIndex >= ENVELOPE_LENGTH-1)
      envelopeIndex = ENVELOPE_LENGTH-2;
      
    // attack phase
    if (noteStrikeTime[noteNo] > noteReleaseTime[noteNo]) {
      envelopeAmplitude = float(envelopeIndexFractional*attackEnvelope[envelopeIndex+1] + (1-envelopeIndexFractional)*attackEnvelope[envelopeIndex]); //lerp
    }
    // release phase
    else if (noteStrikeTime[noteNo] < noteReleaseTime[noteNo]) {
      envelopeAmplitude = float(envelopeIndexFractional*releaseEnvelope[envelopeIndex+1] + (1-envelopeIndexFractional)*releaseEnvelope[envelopeIndex]); //lerp
    }

    // the note volume is given, and envelopeAmplitude is applied as well
    noteVol = noteVolume[noteNo] * envelopeAmplitude;
    // if notevol is zero, continue
    if(noteVol < 0.01)
      continue;
    
    thisIndexFloat = noteBaseIndex[noteNo] + sampleNo * pitchFactor[noteNo];
    
    // clamp index at the end of the sample
    if(thisIndexFloat >= SAMPLE_MAX_SAMPLE_COUNT-2)
      thisIndexFloat = SAMPLE_MAX_SAMPLE_COUNT-2;
    
    thisIndexFract = fract(thisIndexFloat);
    thisIndexUint  = uint(thisIndexFloat) + SAMPLE_MAX_SAMPLE_COUNT*midiNoteNo[noteNo];
    nextIndex = thisIndexUint + 1;
    
    thisSample = sampleBuffer[thisIndexUint];
    nextSample = sampleBuffer[nextIndex];
    
    uint percussiveIndex = uint(secondsSinceStrike*SAMPLE_FREQUENCY);
    if(percussiveIndex >= SAMPLE_MAX_SAMPLE_COUNT)
      percussiveIndex = SAMPLE_MAX_SAMPLE_COUNT-1;
    
    float64_t percussive = sampleBufferPercussive[percussiveIndex];
    innersum = float(thisSample*(1 - thisIndexFract) + nextSample*thisIndexFract + percussive);
    shadersum += innersum * noteVol;
  }

  pcmBufferOut[sampleNo*SHADERS_PER_SAMPLE+shaderIndexInSample] = shadersum / (POLYPHONY / OVERVOLUME);
}
