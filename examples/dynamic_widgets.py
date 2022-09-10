"""
Kivy example for CP1404/CP5632, IT@JCU
Dynamically create buttons based on content of dictionary
Lindsay Ward
Modified from popup_demo, 11/07/2016
"""

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.button import Button
from kivy.properties import StringProperty
from kivy.graphics import Line
from kivy.uix.floatlayout import FloatLayout


class DynamicWidgetsApp(FloatLayout):
    """Main program - Kivy app to demo dynamic widget creation."""

    status_text = StringProperty()

    def __init__(self, **kwargs):
        """Construct main app."""
        super().__init__(**kwargs)
        self.points = []

    def build(self):
        """Build the Kivy GUI."""
        self.title = "Dynamic Widgets"
        self.root = Builder.load_file("dynamic_widgets.kv")
        self.create_widgets()
        return self.root

    def create_widgets(self):
        self.root.ids.entries_box.draw(Line(circle=(150, 150, 50)))

    def press_entry(self, instance):
        """
        Handle pressing entry buttons.
        :param instance: the Kivy button instance that was clicked
        """
        # get name (dictionary key) from the text of Button we clicked on
        name = instance.text
        # update status text
        self.status_text = "{}'s number is {}".format(name, self.name_to_phone[name])

    def clear_all(self):
        """Clear all of the widgets that are children of the "entries_box" layout widget."""
        self.root.ids.entries_box.clear_widgets()

    def on_start(self):
        self.profile = cProfile.Profile()
        self.profile.enable()

    def on_stop(self):
        self.profile.disable()
        self.profile.dump_stats("myapp.profile")

    def on_touch_down(self, touch):
        if super(DynamicWidgetsApp, self).on_touch_down(touch):
            return True
        touch.grab(self)
        self.points.append(touch.pos)
        return True

    def on_touch_move(self, touch):
        if touch.grab_current is self:
            self.points[-1] = touch.pos
            return True
        return super(DynamicWidgetsApp, self).on_touch_move(touch)

    def on_touch_up(self, touch):
        if touch.grab_current is self:
            touch.ungrab(self)
            return True
        return super(DynamicWidgetsApp, self).on_touch_up(touch)

    # event for double-tap
    def on_double_tap(self, instance, *args):
        print("wow!! you have double clicked an image named ")


class testDWA(App):
    def build(self):
        return DynamicWidgetsApp()


if __name__ == "__main__":
    testDWA().run()
