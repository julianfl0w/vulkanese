import os

os.system("export XAUTHORITY=/home/pi/.Xauthority")
os.system("export DISPLAY=:0")

import struct
from bitarray import bitarray
import logging
import json
import sys
import numpy as np
import time
import rtmidi
from rtmidi.midiutil import *
import mido
import math
import hjson as json
import socket
import traceback
import pickle

useMouse = False


class MidiManager:
    def checkForNewDevices(self):
        midi_ports = self.midiin.get_ports()
        for i, midi_portname in enumerate(midi_ports):
            if midi_portname not in self.midi_ports_last:
                logger.debug("adding " + midi_portname)
                try:
                    mididev, midi_portno = open_midiinput(midi_portname)
                except (EOFError, KeyboardInterrupt):
                    sys.exit()

                midiDevAndPatches = (mididev, [self.GLOBAL_DEFAULT_PATCH])
                self.allMidiDevicesAndPatches += [midiDevAndPatches]
        self.midi_ports_last = midi_ports

    def checkKeyboard(self):
        if not keyQueue.empty():
            keyDict = keyQueue.get()
            key = keyDict["name"]
            if key in qwerty2midi.keys():
                if keyDict["event_type"] == "down":
                    msg = mido.Message("note_on", note=qwerty2midi[key], velocity=120)
                else:
                    msg = mido.Message("note_off", note=qwerty2midi[key], velocity=0)

                for dev, patches in self.allMidiDevicesAndPatches:
                    for patch in patches:
                        patch.midi2commands(msg)

    def checkMidi(self, processors):

        for dev, patches in self.allMidiDevicesAndPatches:
            msg = dev.get_message()
            msgs = []
            while msg is not None:
                msgs += [msg]
                msg = dev.get_message()

            processedPW = False
            processedAT = False
            for msg in reversed(msgs):  # most recent first

                msg, dtime = msg
                msg = mido.Message.from_bytes(msg)
                if msg.type == "pitchwheel":
                    if processedPW:
                        continue
                    else:
                        processedPW = True

                if msg.type == "aftertouch":
                    if processedAT:
                        continue
                    else:
                        processedAT = True

                logger.debug(msg)
                for p in processors:
                    p.midi2commands(msg)

    def eventLoop(self):

        # check for new devices once a second
        if time.time() - lastDevCheck > 1:
            lastDevCheck = time.time()
            self.checkForNewDevices()

        # c = sys.stdin.read(1)
        # if c == 'd':
        # 	dtfm_inst.dumpState()
        self.checkMidi()

        if useKeyboard:
            self.checkKeyboard()

        # process the IRQUEUE
        while GPIO.input(37):
            voiceno, opnos = dtfm.getIRQueue()
            self.GLOBAL_DEFAULT_PATCH.processIRQueue(voiceno, opnos)

        if time.time() - lastPatchCheck > 0.02:
            lastPatchCheck = time.time()
            self.checkForPatchChange()

        if useMouse:
            print("CHECKING MOUSE")
            mouseX, mouseY = mouse.get_position()
            # mouseX /= pyautogui.size()[0]
            # mouseY /= pyautogui.size()[1]
            mouseX /= 480
            mouseY /= 360
            if (mouseX, mouseY) != mousePosLast:
                mousePosLast = (mouseX, mouseY)
                self.GLOBAL_DEFAULT_PATCH.midi2commands(
                    mido.Message(
                        "control_change",
                        control=dtfm.ctrl_tremolo_env,
                        value=int(mouseX * 127),
                    )
                )
                self.GLOBAL_DEFAULT_PATCH.midi2commands(
                    mido.Message(
                        "control_change",
                        control=dtfm.ctrl_vibrato_env,
                        value=int(mouseY * 127),
                    )
                )
                # logger.debug((mouseX, mouseY))

    def __init__(self):

        PID = os.getpid()
        if useKeyboard:
            logger.debug("setting up keyboard")
            keyQueue = queue.Queue()
            keyState = {}

            def print_event_json(event):
                keyDict = json.loads(
                    event.to_json(ensure_ascii=sys.stdout.encoding != "utf-8")
                )
                # protect against repeat delay, for simplicity
                # "xset r off" not working
                if keyState.get(keyDict["name"]) != keyDict["event_type"]:
                    keyState[keyDict["name"]] = keyDict["event_type"]
                    # keyQueue.put(json.dumps(keyDict))
                    keyQueue.put(keyDict)
                # sys.stdout.flush()

            keyboard.hook(print_event_json)

        logger.setLevel(0)
        if len(sys.argv) > 1:
            logger.setLevel(1)

        api = rtmidi.API_UNSPECIFIED
        self.midiin = rtmidi.MidiIn(get_api_from_environment(api))

        # loop related variables
        self.midi_ports_last = []
        self.allMidiDevicesAndPatches = []
