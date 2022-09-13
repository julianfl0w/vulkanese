void main() {
       
  uint timeSlice = gl_GlobalInvocationID.x;
  uint polySlice = gl_GlobalInvocationID.y;

  VARIABLEDECLARATIONS
  
  // current time depends on the sample offset
  currTimeWithSampleOffset = currTime[0];// + float(timeSlice)/SAMPLE_FREQUENCY;
        
  for (uint noteNo = polySlice*POLYPHONY_PER_SHADER; noteNo<(polySlice+1)*POLYPHONY_PER_SHADER; noteNo++){

      // calculate the envelope 
      // time is a float holding seconds (since epoch?)
      // these values are updated in the python loop

      // attack phase
      //secondsSinceStrike  = abs(currTimeWithSampleOffset - noteStrikeTime[noteNo] );
      //secondsSinceRelease = abs(currTimeWithSampleOffset - noteReleaseTime[noteNo]);
      secondsSinceStrike  = currTimeWithSampleOffset - noteStrikeTime[noteNo] ;
      secondsSinceRelease = currTimeWithSampleOffset - noteReleaseTime[noteNo];

      // attack phase
      if(noteStrikeTime[noteNo] > noteReleaseTime[noteNo]){
        envelopeIndexFloat64 = secondsSinceStrike*attackSpeedMultiplier[noteNo];
        // keep the fractional part, for interp
        fractional = fract(envelopeIndexFloat64);
        envelopeIndex = int(envelopeIndexFloat64);

        // if envelope is complete, maintain at the final index
        if(envelopeIndex >= ENVELOPE_LENGTH)
            envelopeAmplitude = attackEnvelope[ENVELOPE_LENGTH-1];
        // otherwise, linear interp the envelope
        else
            envelopeAmplitude = attackEnvelope[envelopeIndex];

      }
      // release phase
      else{
        envelopeIndexFloat64 = secondsSinceRelease*releaseSpeedMultiplier[noteNo];
        // keep the fractional part, for interp
        fractional = fract(envelopeIndexFloat64);
        envelopeIndex = int(envelopeIndexFloat64);
        // if envelope is complete, maintain at the final index
        if(envelopeIndex >= ENVELOPE_LENGTH)
            envelopeAmplitude = releaseEnvelope[ENVELOPE_LENGTH-1];
        // otherwise, linear interp the envelope
        else
            envelopeAmplitude = releaseEnvelope[envelopeIndex];
      }

      // the note volume is given, and envelopeAmplitude is applied as well
      noteVol = noteVolume[noteNo] * envelopeAmplitude;


      increment = noteBaseIncrement[noteNo]*pitchFactor[0];
      phase = noteBasePhase[noteNo] + (timeSlice * increment);

      innersum = 0;
      for (uint partialNo = 0; partialNo<PARTIALS_PER_VOICE; partialNo++)
      {

        thisIncrement   = increment * partialMultiplier[partialNo];

        if(thisIncrement < 3.14){
            indexInFilter = int(thisIncrement*(FILTER_STEPS/(3.14)));
            innersum += partialVolume[partialNo] * \
              sin(float(phase)*partialMultiplier[partialNo]) * \
              freqFilter[indexInFilter];
        }

      }
      shadersum+=innersum*noteVol;
  }

  pcmBufferOut[timeSlice*SHADERS_PER_TIMESLICE + polySlice] = shadersum/(PARTIALS_PER_VOICE*POLYPHONY );
}
