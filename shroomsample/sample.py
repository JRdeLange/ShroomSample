import logging
import time

import mido
import noisereduce as nr
import numpy as np
import pedalboard
import sounddevice as sd
from pedalboard import HighpassFilter, LowpassFilter, NoiseGate, Pedalboard
from pedalboard.io import AudioFile
from upath import UPath

from shroomsample.external_instruments import ExternalInstrument
from shroomsample.note import CHORDS, Chord, Note

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

_denoise_sample_store = None


def peak_normalize(audio: np.ndarray, target_db: float = -1.0) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio
    target_amplitude = 10 ** (target_db / 20)
    return audio * (target_amplitude / peak)


def process_audio(
    audio: np.ndarray,
    sample_rate: int,
    input_device_name: str,  # needed for denoising
    post_process: Pedalboard | None = None,
    normalize: bool = True,
    denoise: bool = True,
    noise_profile_gain_db: float = 12.0,
) -> np.ndarray:
    if denoise:
        global _denoise_sample_store
        if _denoise_sample_store is None:
            noise_profile_path = UPath(f"data/noise_clips/{input_device_name}.wav")
            if not noise_profile_path.exists():
                raise ValueError(
                    f"No noise profile found for {input_device_name} at {noise_profile_path}"
                )
            with AudioFile(str(noise_profile_path), "r") as f:
                _denoise_sample_store = f.read(f.frames)

        noise = _denoise_sample_store * (10 ** (noise_profile_gain_db / 20))
        audio = nr.reduce_noise(
            y=audio,
            y_noise=noise,
            sr=sample_rate,
            stationary=True,
            prop_decrease=1,  # how much noise reduction to apply (0-1)
        )

    if post_process is not None:
        audio = post_process(audio, sample_rate)

    if normalize:
        audio = peak_normalize(audio)

    return audio


def save_audio(
    audio: np.ndarray,
    output_path: UPath,
    sample_rate: int = 48000,
):
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    # if there is a # in the filename, put a _ before it to ensure proper sorting (e.g. C#.wav should come after C.wav)
    output_path = UPath(str(output_path).replace("#", "_#"))

    logging.debug(
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
    noise_profile_gain_db: float = 4.0,
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
        post_process=post_process,
        denoise=denoise,
        noise_profile_gain_db=noise_profile_gain_db,
    )

    save_audio(audio, output_path, sample_rate)


def all_one_note(
    instrument: ExternalInstrument,
    midi_output: mido.ports.BaseOutput,
    output_folder: UPath,
    note_name: str = "C",
):
    """
    Sample all notes of a given name
    """
    instrument.select_voice(midi_output)
    for octave in range(1, 8):
        hard = Note(note_str=f"{note_name}{octave}", velocity=80)
        soft = Note(note_str=f"{note_name}{octave}", velocity=40)

        hard_path = output_folder / f"{octave}_{note_name}_hard.wav"
        soft_path = output_folder / f"{octave}_{note_name}_soft.wav"

        record_sample_hardware(
            midi_output,
            hard.play,
            hard_path,
            input_device_name="Microphone (MPC Sample Audio), Windows WASAPI",
        )
        record_sample_hardware(
            midi_output,
            soft.play,
            soft_path,
            input_device_name="Microphone (MPC Sample Audio), Windows WASAPI",
        )


def all_chords(
    instrument: ExternalInstrument,
    midi_output: mido.ports.BaseOutput,
    output_folder: UPath,
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
        chord_path = output_folder / f"{chord_name}.wav"
        record_sample_hardware(
            midi_output,
            chord.play,
            chord_path,
            input_device_name="Microphone (MPC Sample Audio), Windows WASAPI",
        )
