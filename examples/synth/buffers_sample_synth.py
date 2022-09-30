shaderInputBuffers = [
    {"name": "noteBaseIndex", "type": "float64_t", "dims": ["POLYPHONY"]},
    {"name": "noteVolume", "type": "float", "dims": ["POLYPHONY"]},
    {"name": "noteStrikeTime", "type": "float64_t", "dims": ["POLYPHONY"]},
    {"name": "noteReleaseTime", "type": "float64_t", "dims": ["POLYPHONY"]},
    {"name": "currTime", "type": "float64_t", "dims": ["POLYPHONY"]},
    {"name": "attackEnvelope", "type": "float", "dims": ["ENVELOPE_LENGTH"]},
    {"name": "releaseEnvelope", "type": "float", "dims": ["ENVELOPE_LENGTH"]},
    {"name": "attackSpeedMultiplier", "type": "float64_t", "dims": ["POLYPHONY"]},
    {"name": "releaseSpeedMultiplier", "type": "float64_t", "dims": ["POLYPHONY"]},
    {"name": "pitchFactor", "type": "float64_t", "dims": ["POLYPHONY"]},
    {"name": "midiNoteNo", "type": "uint", "dims": ["POLYPHONY"]},
]
shaderInputBuffersNoDebug = []

debuggableVars = [
    # per timeslice, per polyslice (per shader)
    {
        "name": "currTimeWithSampleOffset",
        "type": "float64_t",
        "dims": ["SAMPLES_PER_DISPATCH", "SHADERS_PER_SAMPLE"],
    },
    {
        "name": "shadersum",
        "type": "float",
        "dims": ["SAMPLES_PER_DISPATCH", "SHADERS_PER_SAMPLE"],
    },
    {
        "name": "envelopeIndexFloat64",
        "type": "float64_t",
        "dims": ["SAMPLES_PER_DISPATCH", "POLYPHONY"],
    },
    {
        "name": "envelopeIndex",
        "type": "int",
        "dims": ["SAMPLES_PER_DISPATCH", "POLYPHONY"],
    },
    {
        "name": "sampleVal",
        "type": "float",
        "dims": ["SAMPLES_PER_DISPATCH", "POLYPHONY"],
    },
    # // per timeslice, per note
    {
        "name": "secondsSinceStrike",
        "type": "float64_t",
        "dims": ["SAMPLES_PER_DISPATCH", "POLYPHONY"],
    },
    {
        "name": "secondsSinceRelease",
        "type": "float64_t",
        "dims": ["SAMPLES_PER_DISPATCH", "POLYPHONY"],
    },
    {
        "name": "envelopeIndexFractional",
        "type": "float64_t",
        "dims": ["SAMPLES_PER_DISPATCH", "POLYPHONY"],
    },
    {
        "name": "indexThisNote",
        "type": "float64_t",
        "dims": ["SAMPLES_PER_DISPATCH", "POLYPHONY"],
    },
    {"name": "noteVol", "type": "float", "dims": ["SAMPLES_PER_DISPATCH", "POLYPHONY"]},
    {
        "name": "increment",
        "type": "float64_t",
        "dims": ["SAMPLES_PER_DISPATCH", "POLYPHONY"],
    },
    {
        "name": "innersum",
        "type": "float",
        "dims": ["SAMPLES_PER_DISPATCH", "POLYPHONY"],
    },
    # // per timeslice, per note, per partial
    {
        "name": "thisIncrement",
        "type": "float64_t",
        "dims": ["SAMPLES_PER_DISPATCH", "POLYPHONY"],
    },
    {
        "name": "indexInFilter",
        "type": "int",
        "dims": ["SAMPLES_PER_DISPATCH", "POLYPHONY"],
    },
    {
        "name": "envelopeAmplitude",
        "type": "float",
        "dims": ["SAMPLES_PER_DISPATCH", "POLYPHONY"],
    },
    {
        "name": "thisIndexFloat",
        "type": "float64_t",
        "dims": ["SAMPLES_PER_DISPATCH", "POLYPHONY"],
    },
    {
        "name": "thisIndexFract",
        "type": "float64_t",
        "dims": ["SAMPLES_PER_DISPATCH", "POLYPHONY"],
    },
    {
        "name": "thisIndexUint",
        "type": "uint",
        "dims": ["SAMPLES_PER_DISPATCH", "POLYPHONY"],
    },
    {
        "name": "nextIndex",
        "type": "uint",
        "dims": ["SAMPLES_PER_DISPATCH", "POLYPHONY"],
    },
    {
        "name": "thisSample",
        "type": "float64_t",
        "dims": ["SAMPLES_PER_DISPATCH", "POLYPHONY"],
    },
    {
        "name": "nextSample",
        "type": "float64_t",
        "dims": ["SAMPLES_PER_DISPATCH", "POLYPHONY"],
    },
]

shaderOutputBuffers = [
    {
        "name": "pcmBufferOut",
        "type": "float",
        "dims": ["SAMPLES_PER_DISPATCH", "SHADERS_PER_SAMPLE"],
    },
    {
        "name": "sampleBuffer",
        "type": "float",
        "dims": ["SAMPLE_SET_COUNT", "MIDI_COUNT", "SAMPLE_MAX_SAMPLE_COUNT"],
    },
    {
        "name": "sampleBufferPercussive",
        "type": "float",
        "dims": ["SAMPLE_SET_COUNT", "SAMPLE_MAX_SAMPLE_COUNT"],
    },
]
