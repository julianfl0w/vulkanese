void main() {
  //precision mediump float;
  //precision mediump float64_t;
  uint sampleNo = gl_LocalInvocationID.x;
  uint shaderIndexInSample = gl_LocalInvocationID.y;
  uint zindex = gl_LocalInvocationID.z;
  VARIABLEDECLARATIONS

  // current time depends on the sample offset
  currTimeWithSampleOffset = currTime[0] + sampleNo / float(SAMPLE_FREQUENCY);
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
    
    // keep the fractional part, for interp
    fractional = fract(envelopeIndexFloat64);
    envelopeIndex = int(envelopeIndexFloat64);

    // if envelope is complete, maintain at the second-to-final index
    if (envelopeIndex >= ENVELOPE_LENGTH-1)
      envelopeIndex = ENVELOPE_LENGTH-2;
      
    // attack phase
    if (noteStrikeTime[noteNo] > noteReleaseTime[noteNo]) {
      envelopeAmplitude = float(fractional*attackEnvelope[envelopeIndex+1] + (1-fractional)*attackEnvelope[envelopeIndex]); //lerp
    }
    // release phase
    else if (noteStrikeTime[noteNo] < noteReleaseTime[noteNo]) {
      envelopeAmplitude = float(fractional*releaseEnvelope[envelopeIndex+1] + (1-fractional)*releaseEnvelope[envelopeIndex]); //lerp
    }

    // the note volume is given, and envelopeAmplitude is applied as well
    noteVol = noteVolume[noteNo] * envelopeAmplitude;
    // if notevol is zero, continue
    if(noteVol < 0.01)
      continue;
    
    increment = noteBaseIncrement[noteNo] * pitchFactor[0];
    basePhaseThisNote = noteBasePhase[noteNo] + (sampleNo * increment);

    innersum = 0;
    // loop over the partials in this note
    for (uint partialNo = 0; partialNo < PARTIALS_PER_VOICE; partialNo++) {

      thisIncrement = increment * partialMultiplier[partialNo];

      if (thisIncrement < 1) {
        indexInFilter = int(thisIncrement * FILTER_STEPS);
        float64_t phase = fract(basePhaseThisNote * partialMultiplier[partialNo]);
        slutIndex = uint(phase * SLUTLEN);
        innersum +=
            float(partialVolume[partialNo] *
            //SLUT[slutIndex] *
            sin(float(2*3.1415926*phase)) *
            freqFilter[indexInFilter]);
      }
    }
    shadersum += innersum * noteVol;
  }

  pcmBufferOut = shadersum / (PARTIALS_PER_VOICE * POLYPHONY / OVERVOLUME);
}
