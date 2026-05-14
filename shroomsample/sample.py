import logging
import time

import mido
import numpy as np
import sounddevice as sd
from pedalboard import Pedalboard
from pedalboard.io import AudioFile
from upath import UPath

from shroomsample.external_instruments import ExternalInstrument
from shroomsample.note import CHORDS, NOTE_NAMES, Chord, Note
from shroomsample.process_audio import process_audio
from shroomsample.constants import AUDIO_EXT

logger = logging.getLogger(__name__)

_denoise_sample_store = None


def save_audio(
    audio: np.ndarray,
    output_path: UPath,
    sample_rate: int = 48000,
):
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    logger.debug(
        f"Saving audio to {output_path} with shape {audio.shape} and sample rate {sample_rate}"
    )
    with AudioFile(str(output_path), "w", sample_rate, audio.shape[0]) as f:
        f.write(audio)


def record_sample_hardware(
    midi_output,
    audio_fn: callable,
    output_path: UPath,
    input_device_name: str,
    tail_duration: float = 1.0,
    pre_roll: float = 0.1,
    post_process: Pedalboard | None = None,
    denoise: bool = True,
    noise_profile_gain_db: float = 15.0,
):
    chunks = []

    def _callback(indata, frames, time_info, status):
        # indata is (frames, channels); store as (channels, frames)
        chunks.append(indata.T.copy())

    sample_rate = sd.query_devices(input_device_name, "input")["default_samplerate"]

    with sd.InputStream(
        device=input_device_name,
        channels=2,
        samplerate=sample_rate,
        callback=_callback,
    ):
        time.sleep(pre_roll)  # let buffer stabilize, avoid clipping attack
        audio_fn(midi_output)  # blocks: note_on → sleep(duration) → note_off
        time.sleep(tail_duration)  # capture decay/reverb

    audio = np.concatenate(chunks, axis=1)  # shape: (2, total_samples)
    # take out pre_roll from the beginning
    pre_roll_samples = int(pre_roll * sample_rate)
    audio = audio[:, pre_roll_samples:]

    audio = process_audio(
        audio,
        sample_rate,
        input_device_name,
        normalize=True,
        denoise=denoise,
        noise_profile_gain_db=noise_profile_gain_db,
        post_process=post_process,
    )

    save_audio(audio, output_path, sample_rate)


def all_one_note(
    instrument: ExternalInstrument,
    midi_output: mido.ports.BaseOutput,
    output_folder: UPath,
    input_device_name: str,
    note_name: str,
    octaves: list[int],
):
    """
    Sample all notes of a given name
    """
    instrument.select_voice(midi_output)
    for octave in octaves:
        hard = Note(note_str=f"{note_name}{octave}", velocity=80)
        soft = Note(note_str=f"{note_name}{octave}", velocity=40)

        hard_path = output_folder / str(octave) / f"{note_name}_{octave}_hard{AUDIO_EXT}"
        soft_path = output_folder / str(octave) / f"{note_name}_{octave}_soft{AUDIO_EXT}"

        record_sample_hardware(
            midi_output,
            hard.play,
            hard_path,
            input_device_name=input_device_name,
        )
        record_sample_hardware(
            midi_output,
            soft.play,
            soft_path,
            input_device_name=input_device_name,
        )


def all_notes(
    instrument: ExternalInstrument,
    midi_output: mido.ports.BaseOutput,
    output_folder: UPath,
    input_device_name: str,
    octaves: list[int],
):
    """
    Sample all 12 notes across octaves 1-7 at both hard and soft velocities.
    Files are saved to output_folder/{note_name}/{note_name}_{octave}_{hard|soft}{AUDIO_EXT}
    """
    for note_name in NOTE_NAMES:
        logging.info(f"Recording {note_name} notes...")
        all_one_note(
            instrument,
            midi_output,
            output_folder,
            input_device_name,
            note_name,
            octaves,
        )


def all_chords(
    instrument: ExternalInstrument,
    midi_output: mido.ports.BaseOutput,
    output_folder: UPath,
    input_device_name: str,
    octave: int = 4,
    arp_time: float = 0.0,
    duration: float = 5.0,
    octave_correction: bool = True,
):
    """
    Sample all chords

    octave_correction: if True, notes A, A#, B will be sampled an octave lower
    """
    instrument.select_voice(midi_output)
    for chord_name, notes in CHORDS.items():
        corrected_octave = octave
        if octave_correction and chord_name.startswith(("A", "B")):
            corrected_octave -= 1
        chord = Chord.from_str(
            chord_name, corrected_octave, arp_time=arp_time, duration=duration
        )
        chord_path = output_folder / f"{chord_name}{AUDIO_EXT}"
        record_sample_hardware(
            midi_output,
            chord.play,
            chord_path,
            input_device_name,
        )


