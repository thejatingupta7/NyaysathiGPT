# Nyaysathi-GPT
This repository contains the code for a chatbot using `Langchain` and backed by `Meta-Llama-3-8B` by `Meta` using `Huggingface` and `Streamlit` for frontend UI.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)

## Installation
1. Clone the repository
```
git clone https://github.com/thejatingupta7/NyaysathiGPT
```
2. Create a virtual environment
```
python3 -m venv venv
```
3. Activate the virtual environment
```
source venv/bin/activate
```
4. Install the requirements
```
pip install -r requirements.txt
```

## Usage
1. Embedding Knowledge into the Chatbot.
    - Add your pdfs to the `data_law` folder.
    - Run the following command to ingest the knowledge into the chatbot.
    ```
    python embed.py
    ```
    - This will create a `embed_db/` folder which will contain the embeddings of PDFs i.e. the vectorized information.
2. Running the Chatbot.
    - Run the following command to start the chatbot.
    ```
    python app.py
    ```
