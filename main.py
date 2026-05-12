import logging
import time

import mido
from upath import UPath

from shroomsample.external_instruments import psr270_instruments
from shroomsample.note import CHORDS
from shroomsample.sample import all_chords, all_one_note

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
    start_time = time.time()

    instruments = [psr270_instruments[n] for n in [1, 13, 33]]
    notes = ["C", "F"]
    chord_octaves = range(3, 5)
    arp_times = [0.0, 0.1, 0.3]

    total_samples = len(instruments) * (
        len(notes) * 7 + len(chord_octaves) * len(arp_times) * len(CHORDS)
    )
    logging.info(f"Total samples to be recorded: {total_samples}")

    recorded_samples = 0
    with mido.open_output("U2MIDI Pro 1") as midi_out:
        for instrument in instruments:
            instrument.select_voice(midi_out)
            logging.info(f"Selected instrument: {instrument.name}")
            logging.info(f"Recording notes for {instrument.name}...")
            all_one_note(
                instrument,
                midi_out,
                UPath(f"output/PSR270/{instrument.name}/notes"),
                "C",
            )
            recorded_samples += 7
            progress_bar(recorded_samples, total_samples, start_time)
            all_one_note(
                instrument,
                midi_out,
                UPath(f"output/PSR270/{instrument.name}/notes"),
                "F",
            )
            recorded_samples += 7
            progress_bar(recorded_samples, total_samples, start_time)

            logging.info(f"Recording chords for {instrument.name}...")
            for chord_octave in range(2, 7):
                logging.info(f"Recording chords with root octave {chord_octave}...")
                for arp_time in [0.0, 0.05, 0.2]:
                    logging.info(f"Recording chords with arp_time {arp_time}...")
                    all_chords(
                        instrument,
                        midi_out,
                        UPath(
                            f"output/PSR270/{instrument.name}/chords/octave_{chord_octave}_arp_{arp_time}"
                        ),
                        octave=chord_octave,
                        arp_time=arp_time,
                    )
                    recorded_samples += len(CHORDS)
                    progress_bar(recorded_samples, total_samples, start_time)


if __name__ == "__main__":
    main()
