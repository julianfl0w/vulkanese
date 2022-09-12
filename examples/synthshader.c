void main() {
       
  uint polySlice = gl_GlobalInvocationID.x;
  uint timeSlice = gl_GlobalInvocationID.y;

  float sum = 0;
  
  for (uint noteNo = polySlice*POLYPHONY_PER_SHADER; noteNo<(polySlice+1)*POLYPHONY_PER_SHADER; noteNo++){

      // calculate the envelope 
      // time is a float holding seconds (since epoch?)
      // these values are updated in the python loop
      float env = 0;
      int envelopeIndex;
      float64_t envelopeIndexFloat64;

      // attack phase
      float64_t secondsSinceStrike  = abs(currTime[0] - noteStrikeTime[noteNo] );
      float64_t secondsSinceRelease = abs(currTime[0] - noteReleaseTime[noteNo]);

      // attack phase
      if(noteStrikeTime[noteNo] > noteReleaseTime[noteNo]){
        envelopeIndexFloat64 = secondsSinceStrike*attackSpeedMultiplier[noteNo];
        // keep the fractional part, for interp
        float64_t fractional = fract(envelopeIndexFloat64);
        envelopeIndex = int(envelopeIndexFloat64);

        // if envelope is complete, maintain at the final index
        if(envelopeIndex >= ENVELOPE_LENGTH)
            env = attackEnvelope[ENVELOPE_LENGTH-1];
        // otherwise, linear interp the envelope
        else
            env = attackEnvelope[envelopeIndex];

      }
      // release phase
      else{
        envelopeIndexFloat64 = secondsSinceRelease*releaseSpeedMultiplier[noteNo];
        // keep the fractional part, for interp
        float64_t fractional = fract(envelopeIndexFloat64);
        envelopeIndex = int(envelopeIndexFloat64);
        // if envelope is complete, maintain at the final index
        if(envelopeIndex >= ENVELOPE_LENGTH)
            env = releaseEnvelope[ENVELOPE_LENGTH-1];
        // otherwise, linear interp the envelope
        else
            env = releaseEnvelope[envelopeIndex];
      }

      // the note volume is given, and env is applied as well
      float noteVol = noteVolume[noteNo] * env;


      float increment = noteBaseIncrement[noteNo]*pitchFactor[0];
      float64_t phase = noteBasePhase[noteNo] + (timeSlice * increment);

      float innersum = 0;
      for (uint partialNo = 0; partialNo<PARTIALS_PER_VOICE; partialNo++)
      {
        float vol = partialVolume[partialNo];

        float harmonicRatio   = partialMultiplier[partialNo];
        float thisIncrement = increment * harmonicRatio;

        if(thisIncrement < 3.14){
            int indexInFilter = int(thisIncrement*(FILTER_STEPS/(3.14)));
            innersum += vol * sin(float(phase)*harmonicRatio) * freqFilter[indexInFilter];
        }

      }
      sum+=innersum*noteVol;
  }

  pcmBufferOut[timeSlice*SHADERS_PER_TIMESLICE + polySlice] = sum/(PARTIALS_PER_VOICE*POLYPHONY );
}
