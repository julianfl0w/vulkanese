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

logger = logging.getLogger("dtfm")
# formatter = logging.Formatter('{"debug": %(asctime)s {%(pathname)s:%(lineno)d} %(message)s}')
formatter = logging.Formatter("{{%(pathname)s:%(lineno)d %(message)s}")
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)


def noteToFreq(note):
    a = 440.0  # frequency of A (coomon value is 440Hz)
    return (a / 32) * (2 ** ((note - 9) / 12.0))


class Note:
    def __init__(self, index):
        self.index = index
        self.voices = []
        self.velocity = 0
        self.velocityReal = 0
        self.held = False
        self.polytouch = 0
        self.midiIndex = 0
        self.msg = None
        self.releaseTime = -index
        self.strikeTime = -index


class MidiManager:
    def __init__(self, synthInterface, polyphony):
        self.synthInterface = synthInterface
        PID = os.getpid()

        logger.setLevel(0)
        if len(sys.argv) > 1:
            logger.setLevel(1)

        api = rtmidi.API_UNSPECIFIED
        self.midiin = rtmidi.MidiIn(get_api_from_environment(api))

        # loop related variables
        self.midi_ports_last = []
        self.allMidiDevices = []
        self.lastDevCheck = 0

        # self.flushMidi()

        self.allNotes = [Note(index=i) for i in range(polyphony)]
        self.unheldNotes = self.allNotes.copy()
        self.sustain = False
        self.toRelease = []
        self.modWheelReal = 0.25
        self.pitchwheelReal = 1

    def spawnVoice(self):
        # try to pick an unheld note first
        # the one released the longest ago
        if len(self.unheldNotes):
            retval = sorted(
                self.unheldNotes, key=lambda x: x.strikeTime, reverse=False
            )[0]
            self.unheldNotes.remove(retval)
            return retval
        # otherwise, pick the least recently struck
        else:
            retval = sorted(self.allNotes, key=lambda x: x.strikeTime, reverse=False)[0]
            return retval

    def getNoteFromMidi(self, num):
        for n in self.allNotes:
            if n.midiIndex == num:
                return n
        return self.allNotes[0]

    def checkForNewDevices(self):
        midi_ports = self.midiin.get_ports()
        for i, midi_portname in enumerate(midi_ports):
            if midi_portname not in self.midi_ports_last:
                logger.debug("adding " + midi_portname)
                try:
                    mididev, midi_portno = open_midiinput(midi_portname)
                except (EOFError, KeyboardInterrupt):
                    sys.exit()

                self.allMidiDevices += [mididev]
        self.midi_ports_last = midi_ports

    def processMidi(self, msg):
        print(msg)
        if msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):


            note = self.getNoteFromMidi(msg.note)
            self.unheldNotes += [note]
            if self.sustain:
                self.toRelease += [note]
                return
            note.velocity = 0
            note.velocityReal = 0
            note.midiIndex = -1
            note.held = False
            note.releaseTime = time.time()
            self.synthInterface.noteOff(note)

        elif msg.type == "note_on":
            note = self.spawnVoice()
            note.strikeTime = time.time()
            note.velocity = msg.velocity
            note.velocityReal = (msg.velocity / 127.0) ** 2
            note.held = True
            note.msg = msg
            note.midiIndex = msg.note
            self.synthInterface.noteOn(note)

        elif msg.type == "pitchwheel":
            # print("PW: " + str(msg.pitch))
            self.pitchwheel = msg.pitch
            ARTIPHON = 1
            if ARTIPHON:
                self.pitchwheel *= 12
            amountchange = self.pitchwheel / 8192.0
            octavecount = 2 / 12
            self.pitchwheelReal = pow(2, amountchange * octavecount)
            # print("PWREAL " + str(self.pitchwheelReal))
            # self.setAllIncrements()
            self.synthInterface.pitchWheel(self.pitchwheelReal)

        elif msg.type == "control_change":

            event = "control[" + str(msg.control) + "]"

            # print(event)
            # sustain pedal
            if msg.control == 64:
                print(msg.value)
                if msg.value:
                    self.sustain = True
                else:
                    self.sustain = False
                    for note in self.toRelease:
                        self.synthInterface.noteOff(note)
                    self.toRelease = []

            # mod wheel
            elif msg.control == 1:
                valReal = msg.value / 127.0
                print(valReal)
                self.modWheelReal = valReal
                self.synthInterface.modWheel(self.modWheelReal)

        elif msg.type == "polytouch":
            self.polytouch = msg.value
            self.polytouchReal = msg.value / 127.0

        elif msg.type == "aftertouch":
            self.aftertouch = msg.value
            self.aftertouchReal = msg.value / 127.0

        # if msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
        #    # implement rising mono rate
        #    for heldnote in self.allNotes[::-1]:
        #        if heldnote.held and self.polyphony == self.voicesPerCluster :
        #            self.midi2commands(heldnote.msg)
        #            break

    def flushMidi(self):
        self.getNewMidi()

    def eventLoop(self, processor):

        # check for new devices once a second
        if time.time() - self.lastDevCheck > 1:
            self.lastDevCheck = time.time()
            self.checkForNewDevices()

        for msg in self.getNewMidi():
            self.processMidi(msg)

    def getNewMidi(self):
        # c = sys.stdin.read(1)
        # if c == 'd':
        # 	dtfm_inst.dumpState()
        msgs = []
        for dev in self.allMidiDevices:
            msg = dev.get_message()
            while msg is not None:
                msg = mido.Message.from_bytes(msg[0])
                msgs += [msg]
                msg = dev.get_message()
        return msgs
