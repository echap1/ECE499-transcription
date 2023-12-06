from dataclasses import dataclass

import sounddevice as sd
import numpy as np
from transformers import pipeline
import queue
import sys

whisper = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en")

SAMPLE_RATE = 16000
CHUNK_SIZE = SAMPLE_RATE
CHUNK_BUFFER_SIZE = 5
CHUNK_BUFFER_LAG = 2

audio_queue = queue.Queue()


@dataclass
class Chunk:
    data: np.ndarray

    def __str__(self):
        # Super inefficient but what can ya do
        return f"Chunk({hex(hash(self.data.dumps()))[2:]})"

    __repr__ = __str__


def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(Chunk(indata.copy().flatten()))


def transcribe_stream():
    chunks: list[Chunk] = []

    while True:
        chunks.append(audio_queue.get())
        if len(chunks) > CHUNK_BUFFER_SIZE:
            chunks.pop(0)

        selected_chunk = min(CHUNK_BUFFER_LAG, len(chunks) - 1)

        # print(f"queue={chunks}, selected idx {selected_chunk}")

        data = np.concatenate(tuple(i.data for i in chunks))

        stride_left = CHUNK_SIZE * selected_chunk
        stride_right = CHUNK_SIZE * (len(chunks) - 1 - selected_chunk)

        # print(f"stride: {stride_left}-{stride_right}")

        audio_dict = {
            "sampling_rate": SAMPLE_RATE,
            "raw": data,
            "stride": (stride_left, stride_right)
        }

        try:
            transcription = whisper(audio_dict)
            print(transcription['text'], end='')
        except Exception as e:
            print("Error in transcription:", e)


stream = sd.InputStream(callback=callback, channels=1, samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE)

with stream:
    transcribe_stream()
