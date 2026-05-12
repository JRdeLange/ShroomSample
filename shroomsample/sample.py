import logging
import time

import mido
import noisereduce as nr
import numpy as np
import sounddevice as sd
from pedalboard import Pedalboard
from pedalboard.io import AudioFile
from upath import UPath

from shroomsample.external_instruments import ExternalInstrument
from shroomsample.note import CHORDS, NOTE_NAMES, Chord, Note

logger = logging.getLogger(__name__)

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
    normalize: bool,
    denoise: bool,
    noise_profile_gain_db: float,
    post_process: Pedalboard | None = None,
) -> np.ndarray:
    if denoise:
        global _denoise_sample_store
        if _denoise_sample_store is None:
            noise_profile_path = UPath(f"data/noise_clips/{input_device_name}.flac")
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

        hard_path = output_folder / str(octave) / f"{note_name}_{octave}_hard.flac"
        soft_path = output_folder / str(octave) / f"{note_name}_{octave}_soft.flac"

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
    Files are saved to output_folder/{note_name}/{note_name}_{octave}_{hard|soft}.flac
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
        chord_path = output_folder / f"{chord_name}.flac"
        record_sample_hardware(
            midi_output,
            chord.play,
            chord_path,
            input_device_name="Microphone (MPC Sample Audio), Windows WASAPI",
        )


def mix_audio_files(
    tracks: list[tuple[np.ndarray, float]],
    sample_rate: int,
) -> np.ndarray:
    """
    Mix a list of (audio, offset_seconds) pairs into a single array.
    Shorter tracks are zero-padded at the end. No silence is added at the front.
    """
    total_samples = max(
        audio.shape[1] + int(offset * sample_rate) for audio, offset in tracks
    )
    n_channels = tracks[0][0].shape[0]
    mixed = np.zeros((n_channels, total_samples))
    for audio, offset in tracks:
        start = int(offset * sample_rate)
        mixed[:, start : start + audio.shape[1]] += audio
    return mixed


def synthesize_chord(
    chord: Chord,
    notes_folder: UPath,
    velocity_type: str,
):
    """
    Build a chord by mixing pre-recorded individual note WAVs.
    Arpeggiation is preserved via time offsets matching chord.arp_time.
    velocity_type: "hard" or "soft"
    """
    tracks = []
    sample_rate = None
    for idx, note in enumerate(chord.notes):
        note_name = NOTE_NAMES[note.note % 12]
        octave = (note.note // 12) - 1
        wav_path = str(
            notes_folder / str(octave) / f"{note_name}_{octave}_{velocity_type}.flac"
        )
        with AudioFile(wav_path, "r") as f:
            if sample_rate is None:
                sample_rate = int(f.samplerate)
            audio = f.read(f.frames)
        tracks.append((audio, chord.arp_time * idx))

    mixed = mix_audio_files(tracks, sample_rate)
    mixed = peak_normalize(mixed)

    return mixed, sample_rate


def synthesize_all_chords(
    notes_folder: UPath,
    output_folder: UPath,
    octave: int = 4,
    arp_time: float = 0.0,
    octave_correction: bool = True,
    concat_chords: bool = False,
):
    """
    Synthesize hard and soft variants of all chords by mixing pre-recorded notes.
    Output: output_folder/{chord_name}_hard.flac and output_folder/{chord_name}_soft.flac

    If concat_chords is True, instead of separate files for each chord
    the chord audio will be concatenated into a single file per velocity/major-minor type, saved at output_path/{velocity_type}_chords.flac
    """
    for velocity_type in ("hard", "soft"):
        to_be_concatenated_major = []
        to_be_concatenated_minor = []

        for chord_name in CHORDS:
            corrected_octave = octave
            if octave_correction and chord_name.startswith(("A", "B")):
                corrected_octave -= 1
            chord = Chord.from_str(chord_name, corrected_octave, arp_time=arp_time)
            chord_path = output_folder / f"{chord_name}_{velocity_type}.flac"
            audio, sample_rate = synthesize_chord(chord, notes_folder, velocity_type)

            if not concat_chords:
                save_audio(audio, chord_path, sample_rate)
            else:
                # append to existing file if it exists, otherwise create new file
                if chord_name.endswith("m"):  # minor chord
                    to_be_concatenated_minor.append((audio, sample_rate))
                else:  # major chord
                    to_be_concatenated_major.append((audio, sample_rate))

            logger.debug(f"Synthesized {chord_name} ({velocity_type}) -> {chord_path}")

        if concat_chords:
            for chord_type, tracks in [
                ("major", to_be_concatenated_major),
                ("minor", to_be_concatenated_minor),
            ]:
                if tracks:
                    sample_rate = tracks[0][1]
                    # we need 16 chords for autochop, so we add the last chord until we have 16
                    while len(tracks) < 16:
                        tracks.append(tracks[-1])

                    concatenated = np.concatenate(
                        [audio for audio, _ in tracks], axis=1
                    )

                    concatenated = peak_normalize(concatenated)
                    output_path = (
                        output_folder / f"{velocity_type}_{chord_type}_chords.flac"
                    )
                    save_audio(concatenated, output_path, sample_rate)
                    logger.debug(
                        f"Concatenated {chord_type} chords ({velocity_type}) -> {output_path}"
                    )
