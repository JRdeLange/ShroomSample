import logging

import mido
from upath import UPath

from shroomsample.external_instruments import ExternalInstrument
from shroomsample.note import CHORDS, Chord, Note

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# NOTE: for now without audio capturing and such


def sample(midi_output: mido.ports.BaseOutput, audio_fn: callable, output_path: UPath):
    audio_fn(midi_output)
    logging.debug(f"Saved sample to {output_path}")


def all_cs(
    instrument: ExternalInstrument,
    midi_output: mido.ports.BaseOutput,
    output_folder: UPath,
):
    """
    Sample all C notes
    """
    instrument.select_voice(midi_output)
    for octave in range(1, 8):
        hard = Note(note_str=f"C{octave}", velocity=80)
        soft = Note(note_str=f"C{octave}", velocity=40)

        hard_path = output_folder / f"C{octave}_hard.wav"
        soft_path = output_folder / f"C{octave}_soft.wav"

        sample(midi_output, hard.play, hard_path)
        sample(midi_output, soft.play, soft_path)


def all_chords(
    instrument: ExternalInstrument,
    midi_output: mido.ports.BaseOutput,
    output_folder: UPath,
    octave: int = 4,
    arp_time: float = 0.0,
):
    """
    Sample all chords
    """
    instrument.select_voice(midi_output)
    for chord_name, notes in CHORDS.items():
        chord_notes = [Note(note_str=f"{note}{octave}") for note in notes]
        chord = Chord(notes=chord_notes, arp_time=arp_time)
        chord_path = output_folder / f"{chord_name}{octave}.wav"
        sample(midi_output, chord.play, chord_path)
