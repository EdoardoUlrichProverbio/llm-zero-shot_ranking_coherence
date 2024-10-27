#!/usr/bin/env python3

import os
import requests

def download_file(url, output_path):
    """Downloads a file from a URL to a local path."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(output_path, 'wb') as f:
        print(f"Downloading {output_path} ({total_size / (1024 * 1024):.2f} MB)...")
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                print(f"Downloaded {downloaded / (1024 * 1024):.2f} MB", end='\r')
    print(f"\nCompleted download of {output_path}")

def main():
    model_info_list = [
        {
            'name': 'gptj-6b',
            'url': 'https://huggingface.co/TheBloke/ggml/resolve/main/ggml-gpt-j-6b-q4_0.bin',
            'path': 'llms/models/4bitmodels/ggml-gpt-j-6b-q4_0.bin',
        },
        {
            'name': 'opt-6.7b',
            'url': 'https://huggingface.co/TheBloke/ggml/resolve/main/ggml-opt-6.7b-q4_0.bin',
            'path': 'llms/models/4bitmodels/ggml-opt-6.7b-q4_0.bin',
        },
        {
            'name': 'bloom-7b1',
            'url': 'https://huggingface.co/TheBloke/ggml/resolve/main/ggml-bigscience-bloom-7b1-q4_0.bin',
            'path': 'llms/models/4bitmodels/ggml-bigscience-bloom-7b1-q4_0.bin',
        },
        {
            'name': 'falcon-7b',
            'url': 'https://huggingface.co/TheBloke/ggml/resolve/main/ggml-falcon-7b-instruct-q4_0.bin',
            'path': 'llms/models/4bitmodels/ggml-falcon-7b-instruct-q4_0.bin',
        },
        {
            'name': 'gpt-neo-2.7b',
            'url': 'https://huggingface.co/TheBloke/ggml/resolve/main/ggml-gpt-neo-2.7b-q4_0.bin',
            'path': 'llms/models/4bitmodels/ggml-gpt-neo-2.7b-q4_0.bin',
        },
        {
            'name': 'gpt-neo-1.3b',
            'url': 'https://huggingface.co/TheBloke/ggml/resolve/main/ggml-gpt-neo-1.3b-q4_0.bin',
            'path': 'llms/models/4bitmodels/ggml-gpt-neo-1.3b-q4_0.bin',
        },
        {
            'name': 'gpt2',
            'url': 'https://huggingface.co/TheBloke/ggml/resolve/main/ggml-gpt2-q4_0.bin',
            'path': 'llms/models/4bitmodels/ggml-gpt2-q4_0.bin',
        },
        {
            'name': 'mpt-7b',
            'url': 'https://huggingface.co/TheBloke/MPT-7B-GGML/resolve/main/mpt-7b.ggmlv0.q4_0.bin',
            'path': 'llms/models/4bitmodels/mpt-7b.ggmlv0.q4_0.bin',
        },
        {
            'name': 'wizardlm-7b',
            'url': 'https://huggingface.co/TheBloke/WizardLM-7B-GGML/resolve/main/wizardlm-7b.ggmlv3.q4_0.bin',
            'path': 'llms/models/4bitmodels/wizardlm-7b.ggmlv3.q4_0.bin',
        },
        {
            'name': 'llama-2-7b',
            'url': 'https://huggingface.co/TheBloke/Llama-2-7B-GGML/resolve/main/llama-2-7b.ggmlv3.q4_0.bin',
            'path': 'llms/models/4bitmodels/llama-2-7b.ggmlv3.q4_0.bin',
        },
    ]

    for model_info in model_info_list:
        name = model_info['name']
        url = model_info['url']
        path = model_info['path']

        print(f"Processing {name}...")
        try:
            # Check if file already exists
            if os.path.exists(path):
                print(f"{path} already exists. Skipping download.")
                continue

            download_file(url, path)
            print(f"Downloaded {name} to {path}\n")
        except Exception as e:
            print(f"Failed to download {name}: {e}\n")
            print("Please ensure you have access rights and the URL is correct.")
            print("Some models may require manual download due to licensing restrictions.")

if __name__ == "__main__":
    main()
