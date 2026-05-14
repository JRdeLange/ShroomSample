import numpy as np

from shroomsample.note import CHORDS, NOTE_NAMES, Chord, KEY_SIGNATURES
from shroomsample.process_audio import peak_normalize, find_effective_length
from shroomsample.sample import logger, save_audio
from shroomsample.constants import AUDIO_EXT
from pedalboard.io import AudioFile
from upath import UPath


def length_aware_concatenate(tracks: list[tuple[np.ndarray, float]]) -> np.ndarray:
    """
    Concatenate a list of (audio, sample_rate) pairs into a single array.
    Each track is truncated to its effective length (ignoring trailing silence) before concatenation.
    """
    longest_track = max(find_effective_length(audio) for audio, _ in tracks)
    tracks = [(audio[:, :longest_track], sample_rate) for audio, sample_rate in tracks]

    concatenated = np.concatenate([audio for audio, _ in tracks], axis=1)
    return concatenated


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
            notes_folder / str(octave) / f"{note_name}_{octave}_{velocity_type}{AUDIO_EXT}"
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
    """Synthesize hard and soft variants of all chords by mixing pre-recorded notes.

    Output: output_folder/{chord_name}_hard{AUDIO_EXT} and output_folder/{chord_name}_soft{AUDIO_EXT}

    If concat_chords is True, instead of separate files for each chord the chord
    audio will be concatenated into a single file per velocity/major-minor type,
    saved at output_path/{velocity_type}_chords{AUDIO_EXT}
    """
    for velocity_type in ("hard", "soft"):
        to_be_concatenated_major = []
        to_be_concatenated_minor = []

        for chord_name in CHORDS:
            corrected_octave = octave
            if octave_correction and chord_name.startswith(("A", "B")):
                corrected_octave -= 1
            chord = Chord.from_str(chord_name, corrected_octave, arp_time=arp_time)
            chord_path = output_folder / f"{chord_name}_{velocity_type}{AUDIO_EXT}"
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

                    concatenated = length_aware_concatenate(tracks)
                    concatenated = peak_normalize(concatenated)
                    output_path = (
                        output_folder / f"{velocity_type}_{chord_type}_chords{AUDIO_EXT}"
                    )
                    save_audio(concatenated, output_path, sample_rate)
                    logger.debug(
                        f"Concatenated {chord_type} chords ({velocity_type}) -> {output_path}"
                    )


def synthesize_key_notes(
    notes_folder: UPath,
    output_path: UPath,
    key_signature: str,
    octave: int = 4,
):
    """
    Synthesize two files (hard and soft) that contain notes for two octaves:
    the passed `octave` and the octave above it. Each file contains the notes
    for the given `key_signature` (lookup in `KEY_SIGNATURES`).

    If `output_path` points to a file (has the audio extension), the output
    files are written next to it with `_hard` and `_soft` appended to the
    stem. Otherwise, `output_path` is treated as a folder and files are written
    there named `{key}_{hard/soft}{AUDIO_EXT}`.
    """
    if key_signature not in KEY_SIGNATURES:
        raise KeyError(f"Unknown key signature: {key_signature}")

    notes = KEY_SIGNATURES[key_signature]
    hard_list: list[np.ndarray] = []
    soft_list: list[np.ndarray] = []
    sample_rate = None

    for current_octave in (octave, octave + 1):
        for note_name in notes:
            for velocity, target_list in (("hard", hard_list), ("soft", soft_list)):
                wav_path = str(
                    notes_folder / str(current_octave) / f"{note_name}_{current_octave}_{velocity}{AUDIO_EXT}"
                )
                with AudioFile(wav_path, "r") as f:
                    if sample_rate is None:
                        sample_rate = int(f.samplerate)
                    audio = f.read(f.frames)
                target_list.append(audio)

    # determine output file paths
    if getattr(output_path, "suffix", "") == AUDIO_EXT:
        parent = output_path.parent
        base = output_path.stem
    else:
        parent = output_path
        base = key_signature

    hard_output = parent / f"{base}_hard{AUDIO_EXT}"
    soft_output = parent / f"{base}_soft{AUDIO_EXT}"

    assert len(hard_list) == len(soft_list) == 16, "There should be 16 notes (8 per octave) for each velocity type in a key signature"

    # write hard
    concatenated_hard = length_aware_concatenate([(audio, sample_rate) for audio in hard_list])
    concatenated_hard = peak_normalize(concatenated_hard)
    save_audio(concatenated_hard, hard_output, sample_rate)
    logger.info(f"Wrote key notes (hard, {key_signature}) -> {hard_output}")

    # write soft
    concatenated_soft = length_aware_concatenate([(audio, sample_rate) for audio in soft_list])
    concatenated_soft = peak_normalize(concatenated_soft)
    save_audio(concatenated_soft, soft_output, sample_rate)
    logger.info(f"Wrote key notes (soft, {key_signature}) -> {soft_output}")


def synthesize_key_chords(
    notes_folder: UPath,
    output_folder: UPath,
    key_signature: str,
    arp_time: float = 0.0,
    octave: int = 4,
    octave_correction: bool = True,
):
    """
    For a given `key_signature` (lookup in `KEY_SIGNATURES`) produce two files
    (hard and soft). Each file contains first all major chords built on the
    notes of the key (in the key's note order) and then all corresponding minor
    chords.
    """
    if key_signature not in KEY_SIGNATURES:
        raise KeyError(f"Unknown key signature: {key_signature}")

    notes = KEY_SIGNATURES[key_signature]

    for velocity_type in ("hard", "soft"):
        tracks = []  # will hold (audio, sample_rate) entries

        # first majors
        for note_name in notes:
            chord_name = note_name
            corrected_octave = octave
            if octave_correction and chord_name.startswith(("A", "B")):
                corrected_octave -= 1
            chord = Chord.from_str(chord_name, corrected_octave, arp_time=arp_time)
            audio, sample_rate = synthesize_chord(chord, notes_folder, velocity_type)
            tracks.append((audio, sample_rate))

        # then minors
        for note_name in notes:
            chord_name = f"{note_name}m"
            corrected_octave = octave
            if octave_correction and chord_name.startswith(("A", "B")):
                corrected_octave -= 1
            chord = Chord.from_str(chord_name, corrected_octave, arp_time=arp_time)
            audio, sample_rate = synthesize_chord(chord, notes_folder, velocity_type)
            tracks.append((audio, sample_rate))

        assert len(tracks) == 16, "There should be 16 chords (8 major, 8 minor) for each key signature"

        concatenated = length_aware_concatenate(tracks)
        concatenated = peak_normalize(concatenated)
        output_path = output_folder / f"{key_signature}_{velocity_type}_chords{AUDIO_EXT}"
        save_audio(concatenated, output_path, sample_rate)
        logger.info(f"Wrote key chords ({key_signature}, {velocity_type}) -> {output_path}")
