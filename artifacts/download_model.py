from dotenv import load_dotenv
from huggingface_hub import snapshot_download

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constants import EMBEDDING_MODEL_PATH, TRANSLATION_MODEL_PATH, ENV_PATH

load_dotenv(dotenv_path=ENV_PATH)

def verify_model_files(save_path: str) -> None:
    """
    Verify that all required model files exist in the specified directory.

    :param save_path: Path to the directory where the model is saved.
    :raises FileNotFoundError: If any required file is missing.
    """
    required_files: list[str] = [
        "config.json", 
        "model.safetensors", 
        "config_sentence_transformers.json", 
        "tokenizer.json", 
        "tokenizer_config.json",
        "modules.json",
        "sentence_xlm-roberta_config.json",
        "special_tokens_map.json"
    ]
    existing_files: list[str] = os.listdir(save_path)
    missing_files: list[str] = [file for file in required_files if file not in existing_files]

    if missing_files:
        raise FileNotFoundError(
            f"The following required files are missing in '{save_path}': {', '.join(missing_files)}. "
            f"Ensure the model repository is complete."
        )
    print(f"All required files are present in '{save_path}':")
    for file in existing_files:
        print(file)

def download_model_offline(repo_id: str, save_path: str) -> None:
    """
    Download a Hugging Face model repository locally using snapshot_download.

    :param repo_id: The Hugging Face repository ID (e.g., "codellama/Llama-2-7b-hf").
    :param save_path: The local directory to save the model repository.
    """
    try:
        print(f"Downloading model repository: {repo_id}")
        os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists

        # Download the model repository
        snapshot_download(repo_id=repo_id, repo_type="model", local_dir=save_path)

        # Verify that all required files are downloaded
        verify_model_files(save_path)

        print(f"Model repository saved successfully at: {save_path}")
    except Exception as e:
        print(f"Error downloading the model repository: {e}")


if __name__ == "__main__":
    EMBEDDING_MODEL_PATH.mkdir(parents=True, exist_ok=True)
    TRANSLATION_MODEL_PATH.mkdir(parents=True, exist_ok=True)
    download_model_offline(os.getenv("E_MODEL_NAME"), EMBEDDING_MODEL_PATH)
    download_model_offline(os.getenv("T_MODEL_NAME"), TRANSLATION_MODEL_PATH)