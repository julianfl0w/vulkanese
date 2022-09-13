"""
Line (SmoothLine) Experiment
============================

This demonstrates the experimental and unfinished SmoothLine feature
for fast line drawing. You should see a multi-segment
path at the top of the screen, and sliders and buttons along the bottom.
You can click to add new points to the segment, change the transparency
and width of the line, or hit 'Animate' to see a set of sine and cosine
animations. The Cap and Joint buttons don't work: SmoothLine has not
implemented these features yet.
"""
import os
from math import cos, sin
import numpy as np
import time
import pickle as pkl
import cv2
from scipy import signal
from scipy.interpolate import interp1d
from kivy.uix.floatlayout import FloatLayout

from kivy.app import App
from kivy.properties import (
    OptionProperty,
    NumericProperty,
    StringProperty,
    ListProperty,
    BooleanProperty,
)
from kivymd.uix.behaviors import TouchBehavior
from kivy.lang import Builder
from kivy.clock import Clock


class LinePlayground(FloatLayout):

    releaseLifespan = NumericProperty(0.5)
    attackLifespan = NumericProperty(0.5)
    close = BooleanProperty(False)
    joint = StringProperty("round")
    linewidth = NumericProperty(10.0)
    dt = NumericProperty(0)
    dash_length = NumericProperty(1)
    dash_offset = NumericProperty(0)
    dashes = ListProperty([])
    points = ListProperty([[0, 0], [500, 500]])

    _update_points_animation_ev = None

    def __init__(self, q):

        self.q = q
        Builder.load_string(
            """
<LinePlayground>:
    canvas:
        Color:
            rgba: .4, .4, 1, root.attackLifespan
        Line:
            points: self.points
            joint: self.joint
            width: self.linewidth
            close: self.close
            dash_length: self.dash_length
            dash_offset: self.dash_offset
            dashes: self.dashes

    GridLayout:
        cols: 2
        size_hint: 1, None
        height: 44 * 5

        GridLayout:
            cols: 2

            Label:
                text: 'attackLifespan'
            Slider:
                value: root.attackLifespan
                on_value: root.attackLifespan = float(args[1])
                min: 0.
                max: 1.
            Label:
                text: 'releaseLifespan'
            Slider:
                value: root.releaseLifespan
                on_value: root.releaseLifespan = float(args[1])
                min: 0.
                max: 1.
            Label:
                text: 'Width'
            Slider:
                value: root.linewidth
                on_value: root.linewidth = args[1]
                min: 1
                max: 40

            Label:
                text: 'Close'
            ToggleButton:
                text: 'Close line'
                on_press: root.close = self.state == 'down'

            Label:
                text: 'Dashes'
            GridLayout:
                rows: 1
                ToggleButton:
                    group: 'dashes'
                    text: 'none'
                    state: 'down'
                    allow_no_selection: False
                    size_hint_x: None
                    width: self.texture_size[0]
                    padding_x: '5dp'
                    on_state:
                        if self.state == 'down': root.dashes = []
                        if self.state == 'down': root.dash_length = 1
                        if self.state == 'down': root.dash_offset = 0
                ToggleButton:
                    id: constant
                    group: 'dashes'
                    text: 'Constant: '
                    allow_no_selection: False
                    size_hint_x: None
                    width: self.texture_size[0]
                    padding_x: '5dp'
                    on_state:
                        if self.state == 'down': root.dashes = []
                        if self.state == 'down': root.dash_length = \
                            int(dash_len.text or 1)
                        if self.state == 'down': root.dash_offset = \
                            int(dash_offset.text or 0)
                Label:
                    text: 'len'
                    size_hint_x: None
                    width: self.texture_size[0]
                    padding_x: '5dp'
                TextInput:
                    id: dash_len
                    size_hint_x: None
                    width: '30dp'
                    input_filter: 'int'
                    multiline: False
                    text: '1'
                    on_text: if constant.state == 'down': \
                        root.dash_length = int(self.text or 1)
                Label:
                    text: 'offset'
                    size_hint_x: None
                    width: self.texture_size[0]
                    padding_x: '5dp'
                TextInput:
                    id: dash_offset
                    size_hint_x: None
                    width: '30dp'
                    input_filter: 'int'
                    multiline: False
                    text: '0'
                    on_text: if constant.state == 'down': \
                        root.dash_offset = int(self.text or 0)
                ToggleButton:
                    id: dash_list
                    group: 'dashes'
                    text: 'List: '
                    allow_no_selection: False
                    size_hint_x: None
                    width: self.texture_size[0]
                    padding_x: '5dp'
                    on_state:
                        if self.state == 'down': root.dashes = list(map(lambda\
                            x: int(x or 0), dash_list_in.text.split(',')))
                        if self.state == 'down': root.dash_length = 1
                        if self.state == 'down': root.dash_offset = 0
                TextInput:
                    id: dash_list_in
                    size_hint_x: None
                    width: '180dp'
                    multiline: False
                    text: '4,3,10,15'
                    on_text: if dash_list.state == 'down': root.dashes = \
                        list(map(lambda x: int(x or 0), self.text.split(',')))

        AnchorLayout:
            GridLayout:
                cols: 1
                size_hint: None, None
                size: self.minimum_size
                ToggleButton:
                    size_hint: None, None
                    size: 100, 44
                    text: 'Animate'
                    on_state: root.animate(self.state == 'down')

"""
        )

        FloatLayout.__init__(self)
        print(self.size)

        self.bind(size=self.on_resize)
        self.bind(attackLifespan=self.updateattackLifespan)
        self.bind(releaseLifespan=self.updatereleaseLifespan)

        # self.points = [[0,0], [500,500]]

    def updateattackLifespan(self, obj, value):
        self.q.put(["attackLifespan", value])

    def updatereleaseLifespan(self, obj, value):
        self.q.put(["releaseLifespan", value])

    def points2synth(self):
        # separate xs and ys
        xs = np.array([p[0] for p in self.points])
        ys = np.array([p[1] for p in self.points])
        xs /= np.max(xs)  # normalize on [0,1)
        ys /= np.max(ys)  # normalize on [0,1)

        f1 = interp1d(xs, ys, kind="linear")
        # print(f1)
        # f2 = signal.resample(f1.astype(np.float32), 4*256*64)
        linspac = np.arange(4 * 256) / (4 * 256)
        print(linspac)
        f2 = f1(linspac)
        print(f2)
        self.q.put(["attackEnvelope", np.array(f2, dtype=np.float32)])

    def on_resize(self, obj, size):
        self.points[-1] = [size[0], size[1] / 2]

    def on_touch_down(self, touch):
        if super(LinePlayground, self).on_touch_down(touch):
            return True
        touch.grab(self)
        for i, point in enumerate(self.points.copy()):
            print(abs(touch.pos[0] - point[0]))
            # if this touch is close IN X to an existing point, drag that point

            if abs(touch.pos[0] - point[0]) < 10:
                self.newestPointIndex = i
                # if this is a double tap, remove the point
                if touch.is_double_tap:
                    print("Touch is a double tap !")
                    print(" - interval is", touch.double_tap_time)
                    print(" - distance between previous is", touch.double_tap_distance)
                    del self.points[i]
                self.points2synth()
                return True

        # otherwise, add a new point
        self.points.append(touch.pos)
        self.pointsIndex = sorted(
            range(len(self.points)), key=lambda x: self.points[x][0]
        )
        self.newestPointIndex = self.pointsIndex[-1]
        self.points.sort(key=lambda x: x[0])
        self.points2synth()
        return True

    def on_touch_move(self, touch):
        if touch.grab_current is self:
            # if this is not an edge node
            if (
                self.newestPointIndex > 0
                and self.newestPointIndex < len(self.points) - 1
            ):
                self.points[self.newestPointIndex] = [
                    np.clip(
                        touch.pos[0],
                        self.points[self.newestPointIndex - 1][0] + 10,
                        self.points[self.newestPointIndex + 1][0] - 10,
                    ),
                    touch.pos[1],
                ]

            else:
                self.points[self.newestPointIndex] = [
                    self.points[self.newestPointIndex][0],
                    touch.pos[1],
                ]
            self.points2synth()
            return True
        return super(LinePlayground, self).on_touch_move(touch)

    def on_touch_up(self, touch):
        if touch.grab_current is self:
            touch.ungrab(self)
            return True
        return super(LinePlayground, self).on_touch_up(touch)

    def animate(self, do_animation):
        if do_animation:
            self._update_points_animation_ev = Clock.schedule_interval(
                self.update_points_animation, 0
            )
        elif self._update_points_animation_ev is not None:
            self._update_points_animation_ev.cancel()

    def update_points_animation(self, dt):
        cy = self.height * 0.6
        cx = self.width * 0.1
        w = self.width * 0.8
        step = 20
        points = []
        points2 = []
        self.dt += dt
        for i in range(int(w / step)):
            x = i * step
            points.append(cx + x)
            points.append(cy + cos(x / w * 8.0 + self.dt) * self.height * 0.2)
            points2.append(cx + x)
            points2.append(cy + sin(x / w * 8.0 + self.dt) * self.height * 0.2)
        self.points = points
        self.points2 = points2


class TestLineApp(App):
    def __init__(self, q):
        self.q = q
        App.__init__(self)

    def build(self):
        return LinePlayground(self.q)


def runGui(q):
    TestLineApp(q).run()

    # from synth import runSynth
    # mp.freeze_support()
    # ctx = mp.get_context('spawn')
    # q = ctx.Queue()
    # backendProc = ctx.Process(target=runSynth, args=(q,))
    # backendProc.start()
    # print("PID" + str(backendProc.pid))
    # os.sched_setaffinity(backendProc.pid, {7})
    # print("CPU affinity mask is modified for process id % s" % backendProc.pid)
    # print("Now, process is eligible to run on:", os.sched_getaffinity(backendProc.pid))

    # backendProc.join()


# if __name__ == "__main__":
#     runGui()
#    #frontendProc = ctx.Process(target=runGui, args=(q,))
#    #frontendProc.start()
#    #frontendProc.join()
#    #backendProc.join()
