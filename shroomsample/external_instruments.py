import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

import mido

CONFIGS_PATH = "data/psr_270.json"


@dataclass
class ExternalInstrument(ABC):
    @abstractmethod
    def select_voice(self, midi_out, channel=0):
        pass


@dataclass
class PSR270(ExternalInstrument):
    panel_number: int  # The number shown on the keyboard's physical panel
    name: str  # The name of the instrument
    msb: int  # Bank Select Most Significant Byte (CC 0)
    lsb: int  # Bank Select Least Significant Byte (CC 32)
    program_change: int  # MIDI Program Change number (0-127)

    def select_voice(self, midi_out, channel=0):
        # Send the necessary MIDI messages to select this instrument
        # midi_out.send(
        #     mido.Message("control_change", control=0, value=self.msb, channel=channel)
        # )
        # midi_out.send(
        #     mido.Message("control_change", control=32, value=self.lsb, channel=channel)
        # )
        # midi_out.send(
        #     mido.Message("program_change", program=self.program_change, channel=channel)
        # )
        logging.debug(
            f"Selected instrument {self.name} (Panel {self.panel_number}) with MSB={self.msb}, LSB={self.lsb}, Program Change={self.program_change}"
        )


# Load the instrument configurations from the JSON file
json_data = json.load(open(CONFIGS_PATH))
psr270_instruments = {}

for instrument in json_data["instruments"]:
    psr270_instruments[instrument["panel_number"]] = PSR270(
        panel_number=instrument["panel_number"],
        name=instrument["name"],
        msb=instrument["msb"],
        lsb=instrument["lsb"],
        program_change=instrument["program_change"],
    )
