import speech_recognition as sr
import torch
import datetime
import time
import sounddevice
import numpy as np
from queue import Queue
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available

class AudioTranscriber():
    def __init__(self) -> None:
        # Global variables to store audio setup and transcription
        self.recorder = sr.Recognizer()
        self.source = None
        self.sample_rate = None
        self.audio_model = None
        self.data_queue = Queue()
        self.phrase_time = None
        self.last_sample = bytes()
        self.transcription = ['']
        self.stop_listening_function = None
        self.recording = False

    def setup_audio(self, model_name="medium", energy_threshold=100, sample_rate=16000, default_microphone=None):        
        self.sample_rate = sample_rate
        # Set up Whisper model
        if "large" not in model_name:
            model_name = model_name + ".en"
        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-medium.en", # select checkpoint from https://huggingface.co/openai/whisper-large-v3#model-details
            torch_dtype=torch.float16,
            device="cuda:0", # or mps for Mac devices
            model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
        )
        self.audio_model = pipe
        print("Model loaded.\n")

        # Set up microphone and recognizer
        self.recorder.energy_threshold = energy_threshold
        self.recorder.dynamic_energy_threshold = False

        if default_microphone:
            # Find and use the microphone with the provided name
            mic_index = None
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if default_microphone in name:
                    mic_index = index
                    print(f"Using microphone: {name}")
                    break
            if mic_index is None:
                raise ValueError(f"No microphone found with name containing: {default_microphone}\nAvailable microphones: {sr.Microphone.list_microphone_names()}")
            self.source = sr.Microphone(device_index=mic_index, sample_rate=sample_rate)
        else:
            # Use the default microphone
            self.source = sr.Microphone(sample_rate=sample_rate)

        # Set up the recording in background
        with self.source:
            self.recorder.adjust_for_ambient_noise(self.source)
        
        def record_callback(_, audio:sr.AudioData) -> None:
            if self.recording:
                data = audio.get_raw_data()
                self.data_queue.put(data)

        self.stop_listening_function = self.recorder.listen_in_background(self.source, record_callback, self.phrase_time)

    def transcribe_command(self, record_timeout=0.1, phrase_timeout=0.5):        
        try:
            while True:
                now = datetime.datetime.utcnow()
                if not self.data_queue.empty():
                    phrase_complete = False
                    # If enough time has passed between recordings, consider the phrase complete.
                    # Clear the current working audio buffer to start over with the new data.
                    if self.phrase_time and now - self.phrase_time > datetime.timedelta(seconds=phrase_timeout):
                        self.last_sample = bytes()
                        phrase_complete = True
                    # This is the last time we received new audio data from the queue.
                    self.phrase_time = now

                    # Concatenate our current audio data with the latest audio data.
                    while not self.data_queue.empty():
                        data = self.data_queue.get()
                        self.last_sample += data

                    print("finished recording, begin transcribing...")
                    # Use AudioData to convert the raw data to wav data.
                    audio_data = sr.AudioData(self.last_sample, self.source.SAMPLE_RATE, self.source.SAMPLE_WIDTH)
                    audio_np = audio_data.get_raw_data(
                        convert_rate = 16000,
                    )
                    # Convert raw audio data to numpy array
                    audio_np = np.frombuffer(audio_np, dtype=np.int16).astype(np.float32) / 32768.0

                    # Read the transcription.
                    start_t = time.time()
                    result = self.audio_model(
                        audio_np,
                        chunk_length_s=5,
                        batch_size=1,
                        return_timestamps=True,
                    )
                    print(f"Time taken decode: {time.time() - start_t}")
                    text = result['text'].lower().rstrip(',.;:?!').strip()

                    # Check if the transcribed text is the stop word
                    for stop_word in ["stop", "pardon", "wait"]:
                        if stop_word in text or (self.transcription and stop_word in self.transcription[-1]):
                            return stop_word

                    # If we detected a pause between recordings, add a new item to our transcription.
                    # Otherwise edit the existing one.
                    if phrase_complete:
                        self.transcription.append(text)
                    else:
                        self.transcription[-1] = text

                    # If a phrase was completed, return the transcribed text
                    if self.transcription and self.transcription[-1].strip():
                        return self.transcription[-1].strip()
                    else:
                        return None

                time.sleep(record_timeout)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None

    def stop_stream(self):
        if self.stop_listening_function:
            self.stop_listening_function()
            print("Audio stream stopped.")