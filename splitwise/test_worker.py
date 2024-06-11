import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from functools import partial

import requests
import itertools

POOL = None


PROMPTS = [
    "What is the capital of France?",
    "Explain the theory of relativity in simple terms.",
    "Who wrote 'To Kill a Mockingbird'?",
    "If all bloops are razzies and all razzies are lazzies, are all bloops necessarily lazzies?",
    "What comes next in the sequence: 2, 6, 12, 20, ...?",
    "Solve for x in the equation: 2x + 3 = 7.",
    "What is the integral of x^2?",
    "How are you feeling today?",
    "What do you think about the future of AI?",
    "Write a short story about a robot discovering a hidden talent.",
    "Compose a poem about the changing seasons.",
    "Write a Python function to reverse a string.",
    "Explain the concept of a linked list.",
    "What are the main differences between mitosis and meiosis?",
    "Describe the water cycle.",
    "Who was the first president of the United States?",
    "Explain the causes of World War I.",
    "What is the meaning of life?",
    "Discuss the concept of free will versus determinism.",
    "What is the tallest mountain in the world?",
    "Name three countries that start with the letter 'B'."
]


SAMPLING_PARAMS = {
    "max_tokens": 128,
    "temperature": 0.8,
    "top_p": 0.95,
    "ignore_eos": True
}


def _query_server(prompt: str, host: str) -> dict:
    response = requests.post(host,
                             json={"prompt": prompt, **SAMPLING_PARAMS}, timeout=10)
    response.raise_for_status()
    data = response.json()
    print(data['text'])


def test(num_samples, host):
    prompts = (p for i, p in enumerate(
        itertools.cycle(PROMPTS)) if i < num_samples)
    POOL.map_async(partial(_query_server, host=host), prompts)


if __name__ == "__main__":
    POOL = Pool(8)
    host = "http://localhost:8000/generate"
    num_samples = 10
    test(num_samples, host)
