import noisereduce as nr
import numpy as np

from shroomsample.sample import _denoise_sample_store
from shroomsample.constants import AUDIO_EXT
from pedalboard import Pedalboard
from pedalboard.io import AudioFile
from upath import UPath


def find_effective_length(audio: np.ndarray, threshold: float = 0.01) -> int:
    """
    Find the effective length of an audio signal, ignoring trailing silence.
    """
    # NOTE: test this!!!!!!! Find a good threshold

    energy = np.sum(audio ** 2, axis=0)
    effective_length = np.argmax(energy > threshold)
    return effective_length


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
            noise_profile_path = UPath(f"data/noise_clips/{input_device_name}{AUDIO_EXT}")
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