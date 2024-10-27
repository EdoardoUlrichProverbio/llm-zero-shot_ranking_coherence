import os
import requests
from huggingface_hub import hf_hub_download, HfApi

# Optional: Set your Hugging Face token as an environment variable
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

HUGGINGFACE_TOKEN = "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"

def download_from_huggingface(repo_id, filename, output_path):
    """Downloads the model from Hugging Face Hub."""
    try:
        print(f"Trying Hugging Face Hub for {filename}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Use token if provided
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=os.path.dirname(output_path),
            token=HUGGINGFACE_TOKEN  # Token is optional and used only if available
        )
        # Move the file to the desired output path if necessary
        if file_path != output_path:
            os.rename(file_path, output_path)
        print(f"Downloaded {filename} from Hugging Face to {output_path}")
    except Exception as e:
        print(f"Failed to download {filename} from Hugging Face: {e}")
        print("Please ensure you have access rights and the repository and filename are correct.")


def main():
    model_info_list = [
    {
        'name': 'gptj-6b',
        'repo_id': 'TheBloke/GPT-J-6B-GGML',
        'filename': 'gpt-j-6b.ggmlv3.q4_0.bin',
        'path': 'llms/models/4bitmodels/gpt-j-6b.ggmlv3.q4_0.bin',
    },
    {
        'name': 'opt-6.7b',
        'repo_id': 'TheBloke/OPT-6.7B-GGML',
        'filename': 'opt-6.7b.ggmlv3.q4_0.bin',
        'path': 'llms/models/4bitmodels/opt-6.7b.ggmlv3.q4_0.bin',
    },
    {
        'name': 'bloom-7b1',
        'repo_id': 'TheBloke/bloom-7b1-GGML',
        'filename': 'bloom-7b1.ggmlv3.q4_0.bin',
        'path': 'llms/models/4bitmodels/bloom-7b1.ggmlv3.q4_0.bin',
    },
    {
        'name': 'falcon-7b-instruct',
        'repo_id': 'TheBloke/falcon-7b-instruct-GGML',
        'filename': 'falcon-7b-instruct.ggmlv3.q4_0.bin',
        'path': 'llms/models/4bitmodels/falcon-7b-instruct.ggmlv3.q4_0.bin',
    },
    {
        'name': 'gpt-neo-2.7b',
        'repo_id': 'TheBloke/GPTNeo-2.7B-GGML',
        'filename': 'gptneo-2.7b.ggmlv3.q4_0.bin',
        'path': 'llms/models/4bitmodels/gptneo-2.7b.ggmlv3.q4_0.bin',
    },
    {
        'name': 'gpt-neo-1.3b',
        'repo_id': 'TheBloke/GPTNeo-1.3B-GGML',
        'filename': 'gptneo-1.3b.ggmlv3.q4_0.bin',
        'path': 'llms/models/4bitmodels/gptneo-1.3b.ggmlv3.q4_0.bin',
    },
    {
        'name': 'gpt2',
        'repo_id': 'TheBloke/GPT-2-GGML',
        'filename': 'gpt2.ggmlv3.q4_0.bin',
        'path': 'llms/models/4bitmodels/gpt2.ggmlv3.q4_0.bin',
    },
    {
        'name': 'mpt-7b',
        'repo_id': 'TheBloke/MPT-7B-GGML',
        'filename': 'mpt-7b.ggmlv0.q4_0.bin',
        'path': 'llms/models/4bitmodels/mpt-7b.ggmlv0.q4_0.bin',
    },
    {
        'name': 'wizardlm-7b',
        'repo_id': 'TheBloke/WizardLM-7B-GGML',
        'filename': 'WizardLM-7B.ggmlv3.q4_0.bin',
        'path': 'llms/models/4bitmodels/WizardLM-7B.ggmlv3.q4_0.bin',
    },
    {
        'name': 'llama-2-7b',
        'repo_id': 'TheBloke/Llama-2-7B-GGML',
        'filename': 'llama-2-7b.ggmlv3.q4_0.bin',
        'path': 'llms/models/4bitmodels/llama-2-7b.ggmlv3.q4_0.bin',
    },
]
    for model_info in model_info_list:
        name = model_info['name']
        path = model_info['path']
        repo_id = model_info['repo_id']
        filename = model_info['filename']

        print(f"Processing {name}...")
        try:
            # Check if file already exists
            if os.path.exists(path):
                print(f"{path} already exists. Skipping download.")
                continue

            # Attempt to download from Hugging Face Hub
            download_from_huggingface(repo_id, filename, path)

        except Exception as e:
            print(f"Failed to download {name}: {e}\n")
            print("Please ensure you have access rights and the repository and filename are correct.")
            print("Some models may require manual download due to licensing restrictions.")

if __name__ == "__main__":
    main()