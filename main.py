import logging
import time

import mido
from upath import UPath

from shroomsample.external_instruments import psr270_instruments
from shroomsample.sample import all_notes, synthesize_all_chords

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def progress_bar(progress, total, start_time, bar_length=40):
    """
    Display a progress bar in the console along with an estimated time remaining.
    """
    logging.info(f"Recorded {progress}/{total} samples")

    fraction = progress / total
    filled_length = int(bar_length * fraction)
    bar = "=" * filled_length + "-" * (bar_length - filled_length)
    elapsed_time = time.time() - start_time
    estimated_total_time = elapsed_time / fraction if fraction > 0 else 0
    remaining_time = estimated_total_time - elapsed_time
    logging.info(f"\r[{bar}] {progress}/{total} - ETA: {remaining_time:.2f}s")


def main():
    base_output_path = UPath("output_flac/PSR270")

    start_time = time.time()

    instruments = [psr270_instruments[n] for n in [1]]

    note_octaves = range(2, 8)
    chord_octaves = range(3, 7)

    arp_times = [0.0, 0.03, 0.05, 0.1, 0.2, 0.3]

    # 12 notes × 7 octaves × 2 velocities (hard + soft) per instrument
    total_samples = len(instruments) * 12 * len(note_octaves) * 2
    logging.info(f"Total samples to be recorded: {total_samples}")

    recorded_samples = 0
    with mido.open_output("U2MIDI Pro 1") as midi_out:
        for instrument in instruments:
            logging.info(f"Recording all notes for {instrument.name}...")
            all_notes(
                instrument,
                midi_out,
                base_output_path / f"{instrument.name}/notes",
                "Line In (High Definition Audio Device), Windows WASAPI",
                octaves=note_octaves,
            )
            recorded_samples += 12 * len(note_octaves) * 2
            progress_bar(recorded_samples, total_samples, start_time)

            for chord_octave in chord_octaves:
                for arp_time in arp_times:
                    logging.info(
                        f"Synthesizing chords for {instrument.name}, octave {chord_octave}, arp_time {arp_time}..."
                    )
                    synthesize_all_chords(
                        notes_folder=base_output_path / f"{instrument.name}/notes",
                        output_folder=base_output_path
                        / f"{instrument.name}/chords_concat/octave_{chord_octave}_arp_{arp_time}",
                        octave=chord_octave,
                        arp_time=arp_time,
                        concat_chords=True,
                    )


if __name__ == "__main__":
    main()
