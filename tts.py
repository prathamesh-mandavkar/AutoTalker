from transformers import BarkModel, AutoProcessor, set_seed
import torch
from optimum.bettertransformer import BetterTransformer
import numpy as np
from pydub import AudioSegment
from pedalboard.io import AudioFile
from pedalboard import *
import noisereduce as nr

class AudioProcessor:
    """
    A class for processing audio using the Bark TTS model and applying audio effects.

    Attributes:
        device (str): The device to run the model on, either "cuda:0" if CUDA is available, else "cpu".
        processor (AutoProcessor): The processor for the Bark TTS model.
        model (BarkModel): The Bark TTS model.
        sampling_rate (int): The sampling rate used for audio generation.
    """

    def __init__(self):
        """
        Initializes the AudioProcessor class.
        """
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained("suno/bark-small")
        self.model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16).to(self.device)
        self.model = BetterTransformer.transform(self.model, keep_original_model=False)
        self.model.enable_cpu_offload()
        self.sampling_rate = self.model.generation_config.sample_rate

    def BarkTTS(self, text_prompt, voice_preset):
        """
        Generates audio from text using the Bark TTS model.

        Args:
            text_prompt (str): The text prompt to generate audio from.
            voice_preset (str): The voice preset to use for generation.

        Returns:
            np.ndarray: The generated audio as a numpy array.
        """
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        silence = np.zeros(int(0.25 * self.sampling_rate))
        text_prompt = text_prompt.replace("\n", " ").strip()

        sentences = [sentence.strip() for sentence in text_prompt.split('.') if sentence]

        audio_segments = []
        for segment in sentences:
            inputs = self.processor(segment, return_tensors="pt", voice_preset=voice_preset).to(self.device)
            set_seed(0)
            torch.cuda.synchronize()
            audio_outputs = self.model.generate(**inputs)
            # audio_segments.append(audio_outputs.squeeze().cpu().numpy())
            audio_segments += [audio_outputs.squeeze().cpu().numpy(),silence.copy()]

        torch.cuda.synchronize()

        output = np.concatenate(audio_segments)
        return output

    def enhance_audio(self, input_file, output_file):
        """
        Enhances audio from an input file and saves it to an output file.

        Args:
            input_file (str): The path to the input audio file.
            output_file (str): The path to save the enhanced audio file.
        """
        sr = 44100

        with AudioFile(input_file).resampled_to(sr) as f:
            audio = f.read(f.frames)

        reduced_noise = nr.reduce_noise(y=audio, sr=sr, stationary=True, prop_decrease=0.75)

        board = Pedalboard([
            NoiseGate(threshold_db=-30, ratio=1.5, release_ms=250),
            Compressor(threshold_db=-16, ratio=2.5),
            LowShelfFilter(cutoff_frequency_hz=400, gain_db=10, q=1),
            Gain(gain_db=10)
        ])

        effected = board(reduced_noise, sr)

        with AudioFile(output_file, 'w', sr, effected.shape[0]) as f:
            f.write(effected)

    def save_audio(self, audio_data, output_file, format="mp3"):
        """
        Saves audio data to a file.

        Args:
            audio_data (np.ndarray): The audio data to save.
            output_file (str): The path to save the audio file.
            format (str): The format to save the audio file, defaults to "mp3".
        """
        int_audio_arr = (audio_data * np.iinfo(np.int16).max).astype(np.int16)

        audio_segment = AudioSegment(
            int_audio_arr.tobytes(),
            frame_rate=self.sampling_rate,
            sample_width=int_audio_arr.dtype.itemsize,
            channels=1,
        )
        audio_segment.export(output_file, format=format)

