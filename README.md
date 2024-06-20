# Transforming Scholarly Document Summarization

This repository contains code for my bachelor thesis on scholarly document summarization for the Bachelor of Information Science at the Rijksuniversiteit Groningen. The code includes the following:

1. Filtering and selecting random articles from a BibTeX file.
2. Summarizing scholarly documents using the LLaMA 3 large language model with in-context learning techniques.

All the experiments were ran on the Hábrók server of the Rijksuniversiteit Groningen

## Setup

### Prerequisites

- Python 3.6 or higher
- pip (Python package installer)

### Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/your-username/scholarly-document-summarization.git
    cd scholarly-document-summarization
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Filtering and Selecting Articles

To filter and select articles from a BibTeX file:

1. Unzip the bibtext.zip file form this github to reproduce results, otherwise download a newer bibtext file from https://aclanthology.org/
2. Place your BibTeX file as `bibtext.txt` in the root directory
3. Run the script:

    ```bash
    python rand.py
    ```

This will generate a `data.json` file containing the selected articles.

### Summarizing Articles

To summarize articles using the LLaMA 3 model:

1. Ensure you are logged in to Huggingface using huggingface-cli login and make sure your account has access to the Llama-3-8b-Instruct model.
2. Run the summarization script and make sure the data.json file is in the same directory:

    ```bash
    python summarizer.py
    ```

## Files

- `rand.py`: Script to filter and randomly select articles from a BibTeX file.
- `summarizer.py`: Script to summarize articles using the LLaMA 3 model.
- `requirements.txt`: Required Python packages.


## Contact

For any questions or issues, please contact [e.kort.3@student.rug.nl].
