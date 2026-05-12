import logging
import threading
import time
from dataclasses import dataclass

import mido

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

_midi_send_lock = threading.Lock()

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

CHORDS = {
    "C": ["C", "E", "G"],
    "Cm": ["C", "D#", "G"],
    "C#": ["C#", "F", "G#"],
    "C#m": ["C#", "E", "G#"],
    "D": ["D", "F#", "A"],
    "Dm": ["D", "F", "A"],
    "D#": ["D#", "G", "A#"],
    "D#m": ["D#", "F#", "A#"],
    "E": ["E", "G#", "B"],
    "Em": ["E", "G", "B"],
    "F": ["F", "A", "C"],
    "Fm": ["F", "G#", "C"],
    "F#": ["F#", "A#", "C#"],
    "F#m": ["F#", "A", "C#"],
    "G": ["G", "B", "D"],
    "Gm": ["G", "A#", "D"],
    "G#": ["G#", "C", "D#"],
    "G#m": ["G#", "B", "D#"],
    "A": ["A", "C#", "E"],
    "Am": ["A", "C", "E"],
    "A#": ["A#", "D", "F"],
    "A#m": ["A#", "C#", "F"],
    "B": ["B", "D#", "F#"],
    "Bm": ["B", "D", "F#"],
}


def note_to_midi(name: str) -> int:
    # e.g. "C4", "F#3", "Bb2"
    name = (
        name.replace("Bb", "A#")
        .replace("Eb", "D#")
        .replace("Ab", "G#")
        .replace("Db", "C#")
        .replace("Gb", "F#")
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
        logging.debug(f"Sending MIDI message: {message}")
        # midi_out.send(message)


@dataclass
class Note:
    note: int | None = None  # MIDI note number (0-127)
    note_str: str | None = None  # e.g. "C4", "D#5"
    velocity: int = 64
    channel: int = 0
    duration: float = 1.0

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

    def play(self, midi_out: mido.ports.BaseOutput):
        threads = []
        for note in self.notes:
            t = threading.Thread(target=note.play, args=(midi_out,))
            t.start()
            threads.append(t)
            time.sleep(self.arp_time)

        for t in threads:
            t.join()
