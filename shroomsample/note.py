import logging
import threading
import time
from dataclasses import dataclass

import mido

logger = logging.getLogger(__name__)

_midi_send_lock = threading.Lock()

NOTE_NAMES = ["C", "C_#", "D", "D_#", "E", "F", "F_#", "G", "G_#", "A", "A_#", "B"]

CHORDS = {
    "C": ["C", "E", "G"],
    "Cm": ["C", "D_#", "G"],
    "C_#": ["C_#", "F", "G_#"],
    "C_#m": ["C_#", "E", "G_#"],
    "D": ["D", "F_#", "A"],
    "Dm": ["D", "F", "A"],
    "D_#": ["D_#", "G", "A_#"],
    "D_#m": ["D_#", "F_#", "A_#"],
    "E": ["E", "G_#", "B"],
    "Em": ["E", "G", "B"],
    "F": ["F", "A", "C"],
    "Fm": ["F", "G_#", "C"],
    "F_#": ["F_#", "A_#", "C_#"],
    "F_#m": ["F_#", "A", "C_#"],
    "G": ["G", "B", "D"],
    "Gm": ["G", "A_#", "D"],
    "G_#": ["G_#", "C", "D_#"],
    "G_#m": ["G_#", "B", "D_#"],
    "A": ["A", "C_#", "E"],
    "Am": ["A", "C", "E"],
    "A_#": ["A_#", "D", "F"],
    "A_#m": ["A_#", "C_#", "F"],
    "B": ["B", "D_#", "F_#"],
    "Bm": ["B", "D", "F_#"],
}


KEY_SIGNATURES = {
    "C-Am":    ["C", "D", "E", "F", "G", "A", "B"],
    "C_#-A_#m":["C_#", "D_#", "F", "F_#", "G_#", "A_#", "C"],
    "D-Bm":    ["D", "E", "F_#", "G", "A", "B", "C_#"],
    "D_#-Cm":  ["D_#", "F", "G", "G_#", "A_#", "C", "D"],
    "E-C_#m":  ["E", "F_#", "G_#", "A", "B", "C_#", "D_#"],
    "F-Dm":    ["F", "G", "A", "A_#", "C", "D", "E"],
    "F_#-D_#m":["F_#", "G_#", "A_#", "B", "C_#", "D_#", "F"],
    "G-Em":    ["G", "A", "B", "C", "D", "E", "F_#"],
    "G_#-Fm":  ["G_#", "A_#", "C", "C_#", "D_#", "F", "G"],
    "A-F_#m":  ["A", "B", "C_#", "D", "E", "F_#", "G_#"],
    "A_#-Gm":  ["A_#", "C", "D", "D_#", "F", "G", "A"],
    "B-G_#m":  ["B", "C_#", "D_#", "E", "F_#", "G_#", "A_#"],
}


def note_to_midi(name: str) -> int:
    # e.g. "C4", "F_#3", "Bb2"
    name = (
        name.replace("Bb", "A_#")
        .replace("Eb", "D_#")
        .replace("Ab", "G_#")
        .replace("Db", "C_#")
        .replace("Gb", "F_#")
    )
    octave = int(name[-1])
    note = name[:-1]
    return (octave + 1) * 12 + NOTE_NAMES.index(note)


def midi_to_note(midi: int) -> str:
    octave = (midi // 12) - 1
    note = NOTE_NAMES[midi % 12]
    return f"{note}{octave}"


def _send_midi_message(midi_out: mido.ports.BaseOutput, message: mido.Message):
    with _midi_send_lock:
        logger.debug(f"Sending MIDI message: {message}")
        midi_out.send(message)


@dataclass
class Note:
    note: int | None = None  # MIDI note number (0-127)
    note_str: str | None = None  # e.g. "C4", "D_#5"
    velocity: int = 64
    channel: int = 0
    duration: float = 5.0

    def __post_init__(self):
        if self.note is not None and self.note_str is not None:
            if self.note != note_to_midi(self.note_str):
                raise ValueError("note and note_str do not match")
        elif self.note is None and self.note_str is not None:
            self.note = note_to_midi(self.note_str)
        elif self.note is not None and self.note_str is None:
            self.note_str = midi_to_note(self.note)
        elif self.note is None and self.note_str is None:
            raise ValueError("Either note or note_str must be provided")

    def play(self, midi_out: mido.ports.BaseOutput):
        _send_midi_message(
            midi_out,
            mido.Message(
                "note_on", note=self.note, velocity=self.velocity, channel=self.channel
            ),
        )
        time.sleep(self.duration)
        _send_midi_message(
            midi_out,
            mido.Message(
                "note_off", note=self.note, velocity=self.velocity, channel=self.channel
            ),
        )


@dataclass
class Chord:
    notes: list[Note]
    arp_time: float = 0.0

    @classmethod
    def from_str(
        cls, chord_str: str, octave: int, arp_time: float = 0.0, duration: float = 5.0
    ) -> "Chord":
        """
        Create a Chord from a chord name in CHORDS.
        The root note is placed in the given octave; all other notes are placed
        in the nearest octave above the root (octave+1 if their pitch class is
        below the root's pitch class, otherwise the same octave).
        """
        note_names = CHORDS[chord_str]
        root_note_idx = NOTE_NAMES.index(note_names[0])
        notes = []
        for idx, name in enumerate(note_names):
            note_idx = NOTE_NAMES.index(name.replace("Bb", "A_#").replace("Eb", "D_#"))
            note_octave = octave + 1 if note_idx < root_note_idx else octave
            notes.append(
                Note(
                    note_str=f"{name}{note_octave}", duration=duration - arp_time * idx
                )
            )

        return cls(notes=notes, arp_time=arp_time)

    def play(self, midi_out: mido.ports.BaseOutput):
        threads = []
        for note in self.notes:
            t = threading.Thread(target=note.play, args=(midi_out,))
            t.start()
            threads.append(t)
            time.sleep(self.arp_time)

        for t in threads:
            t.join()
