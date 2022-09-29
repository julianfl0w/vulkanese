import mido
import sounddevice as sd
import numpy as np

# standardized q reader
class JSynth:
    def __init__(self):

        # Start the sound server
        if self.PYSOUND:
            self.stream = sd.Stream(
                samplerate=self.SAMPLE_FREQUENCY,
                blocksize=self.SAMPLES_PER_DISPATCH,
                device=None,
                channels=self.CHANNELS,
                dtype=np.float32,
                latency=self.LATENCY_SECONDS,
                extra_settings=None,
                callback=None,
                finished_callback=None,
                clip_off=None,
                dither_off=None,
                never_drop_input=None,
                prime_output_buffers_using_stream_callback=None,
            )

        if self.GRAPH:
            # to run GUI event loop
            plt.ion()

            # here we are creating sub plots
            self.figure, ax = plt.subplots(figsize=(10, 8))
            self.newVal = np.ones((4 * self.ENVELOPE_LENGTH * self.POLYPHONY))
            (self.plot,) = ax.plot(self.newVal)
            plt.ylabel("some numbers")
            plt.show()
            plt.ylim(-2, 2)

        if self.PYSOUND:
            self.stream.start()

    def range2unity(self, maxi):
        if maxi == 1:
            unity = [0]
        else:
            ar = np.arange(maxi)
            unity = ar - np.mean(ar)
            unity /= max(unity)
        return unity

    def checkQ(self):
        # check the queue (q) for incoming commands
        if self.q is not None:
            if self.q.qsize():
                recvd = self.q.get()
                # print(recvd)
                varName, self.newVal = recvd
                if varName == "attackEnvelope":
                    self.sampleInterface.computePipeline.attackEnvelope.setBuffer(
                        self.newVal
                    )
                elif varName == "attackLifespan":
                    mini = 0.25  # minimum lifespan, seconds
                    maxi = 5  # maximim lifespan, seconds
                    self.newVal = mini + (self.newVal * (maxi - mini))
                    multiplier = self.ENVELOPE_LENGTH / (self.newVal)
                    print(multiplier)
                    self.sampleInterface.computePipeline.attackSpeedMultiplier.setBuffer(
                        self.POLYLEN_ONES64 * multiplier
                    )

                elif varName == "releaseLifespan":
                    mini = 0.25  # minimum lifespan, seconds
                    maxi = 5  # maximim lifespan, seconds
                    self.newVal = mini + (self.newVal * (maxi - mini))
                    multiplier = self.ENVELOPE_LENGTH / (self.newVal)
                    print(multiplier)
                    self.sampleInterface.computePipeline.releaseSpeedMultiplier.setBuffer(
                        self.POLYLEN_ONES64 * multiplier
                    )

                elif varName == "partialSpread":
                    mini = 0.0001  # minimum
                    maxi = 0.001  # maximim
                    self.newVal = mini + (self.newVal * (maxi - mini))
                    self.sampleInterface.PARTIAL_SPREAD = self.newVal
                    self.sampleInterface.updatePartials()

                elif varName == "partialCount":
                    mini = 1  # minimum
                    maxi = 15  # maximim
                    self.newVal = mini + (self.newVal * (maxi - mini))
                    self.sampleInterface.PARTIALS_PER_HARMONIC = int(self.newVal)
                    self.sampleInterface.updatePartials()

    def updatingGraph(self, data):
        # print(pa2)
        # updating data values
        self.plot.set_ydata(data)

        # drawing updated values
        self.figure.canvas.draw()

        # This will run the GUI event
        # loop until all UI events
        # currently waiting have been processed
        self.figure.canvas.flush_events()

    def runTest(self):

        # start middle A note
        self.mm.processMidi(mido.Message("note_on", note=50, velocity=64, time=6.2))
        self.mm.processMidi(mido.Message("note_on", note=60, velocity=64, time=6.2))
        # self.midi2commands(mido.Message("note_on", note=70, velocity=64, time=6.2))

    def updatePitchBend(self):

        # with artiphon, only bend recent note
        ARTIPHON = False
        if ARTIPHON:
            self.POLYLEN_ONES_POST64[:] = self.fullAddArray[:]
            self.POLYLEN_ONES_POST64[
                self.mm.mostRecentlyStruckNoteIndex * 2
            ] = (
                self.POLYLEN_ONES_POST64[
                    self.mm.mostRecentlyStruckNoteIndex * 2
                ]
                * self.mm.pitchwheelReal
            )
            self.postBendArray += self.POLYLEN_ONES_POST64

            self.POLYLEN_ONES_POST64[:] = self.POLYLEN_ONES64[:]
            self.POLYLEN_ONES_POST64[
                self.mm.mostRecentlyStruckNoteIndex * 2
            ] = self.mm.pitchwheelReal
            self.computePipeline.pitchFactor.setBuffer(self.POLYLEN_ONES_POST64)

        else:
            self.postBendArray += self.fullAddArray * self.mm.pitchwheelReal
            self.computePipeline.pitchFactor.setBuffer(
                self.POLYLEN_ONES64 * self.mm.pitchwheelReal
            )
