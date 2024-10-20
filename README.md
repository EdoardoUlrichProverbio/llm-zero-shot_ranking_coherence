# LLM Zero-Shot Ranking Coherence

This project explores whether large language models (LLMs) respect the transitivity property in zero-shot ranking tasks. Specifically, it focuses on ranking sets of movie descriptions by similarity to a specific genre and checking if the rankings maintain logical consistency.

## Project Overview

The repository contains code for:

- Ranking movie descriptions using various LLMs in a zero-shot setting.
- Evaluating transitivity in rankings to identify contradictions.
- Experimenting with multiple models such as `gpt2`, `gpt-neo-2.7B`, `gpt-j-6B`, and others for comparative analysis.

## Setup Instructions

To set up the project locally:

1. **Clone the Repository**

2. **Set Up Environment**
   Run the setup script to create a virtual environment and install dependencies:

3. **Configuration**
   Ensure that a GPU-enabled environment is available for efficient model execution.

## Running the Code

The main script can be executed to perform the ranking tasks:

The script will load the LLMs specified in `model_names` and perform zero-shot ranking on the given dataset.

## Requirements

- Python 3.8+
- GPU (recommended)

Dependencies are listed in the `requirements.txt` file, including:

- `transformers`
- `bitsandbytes`
- `torch`
- `kagglehub`

## LLMs Used

This project uses the following models for ranking:

- `gpt2`
- `EleutherAI/gpt-neo-2.7B`
- `EleutherAI/gpt-j-6B`
- `facebook/opt-6.7b`
- `bigscience/bloom-7b1`
- `tiiuae/falcon-7b`

## Goal

The main goal is to investigate if LLMs produce logically consistent rankings and whether they follow the transitivity property when ranking sets based on similarity.

## License

This project is licensed under the MIT License.

## Contact

For questions or contributions, feel free to open an issue or reach out to [Edoardo Ulrich Proverbio](https://github.com/EdoardoUlrichProverbio).



