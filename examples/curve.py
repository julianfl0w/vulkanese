import scipy.interpolate.lagrange


class curve:
    def __init__(
        self,
        totallength=1024,
        smoothness=1300,
        resonance=4,
        sharpness=200,
        filtclass="low",
    ):
        self.length = length
        self.smoothness = smoothness
        self.sharpness = sharpness
        self.length = length
        self.filtclass = filtclass
        self.calculate()

    def calculate(self):
        # initialize entire shape to ones
        self.curveArray = np.ones((4 * self.length), dtype=np.float32)
        if self.filtclass == "low":
            # remove high end
            self.curveArray[4 * self.length / 2 : 4 * self.length] = 0
        else:
            # remove low end
            self.curveArray[0 : 4 * self.length / 2] = 0

        # put a transition-value in the middle
        self.curveArray[
            4 * self.length / 2 - sharpness : 4 * self.length / 2 + sharpness
        ] = resonance

        # the above steps create a shape like this:

        #                    _________
        #            _______|
        #           |
        # ___________|

        self.curveArray[:] = np.convolve(self.curveArray, np.array([1] * smoothness))[
            smoothness - 1 :
        ]
        self.curveArray /= np.max(self.curveArray)


class lagrangeCurve:
    def __init__(
        self,
        totallength=1024,
        smoothness=1300,
        resonance=4,
        sharpness=200,
        filtclass="low",
    ):
        self.length = length
        self.smoothness = smoothness
        self.sharpness = sharpness
        self.length = length
        self.filtclass = filtclass
        self.calculate()

    def calculate(self):
        # initialize entire shape to ones
        self.curveArray = np.ones((4 * self.length), dtype=np.float32)
        if self.filtclass == "low":
            # remove high end
            self.curveArray[4 * self.length / 2 : 4 * self.length] = 0
        else:
            # remove low end
            self.curveArray[0 : 4 * self.length / 2] = 0

        # put a transition-value in the middle
        self.curveArray[
            4 * self.length / 2 - sharpness : 4 * self.length / 2 + sharpness
        ] = resonance

        # the above steps create a shape like this:

        #                    _________
        #            _______|
        #           |
        # ___________|

        self.curveArray[:] = np.convolve(self.curveArray, np.array([1] * smoothness))[
            smoothness - 1 :
        ]
        self.curveArray /= np.max(self.curveArray)

        lagrange
