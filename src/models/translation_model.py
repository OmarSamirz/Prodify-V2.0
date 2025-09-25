import torch
from langid import classify
from dotenv import load_dotenv
from transformers import MarianMTModel, MarianTokenizer

import os

from constants import DTYPE_MAP, TRANSLATION_MODEL_PATH
from modules.logger import logger

load_dotenv()


class TranslationModel:

    def __init__(self):
        self.model_id = os.getenv("T_MODEL_NAME")
        self.device = torch.device(os.getenv("T_DEVICE"))
        self.truncation = eval(os.getenv("T_TRUNCATION"))
        self.padding = eval(os.getenv("T_PADDING"))
        self.skip_special_tokens = eval(os.getenv("T_SKIP_SPECIAL_TOKENS"))
        dtype = os.getenv("T_DTYPE")
        if dtype in DTYPE_MAP:
            self.dtype = DTYPE_MAP[dtype]
        else:
            raise ValueError(f"This dtype {dtype} is not supported.")

        try:
            self.model = MarianMTModel.from_pretrained(
                TRANSLATION_MODEL_PATH,
                device_map=self.device, 
                torch_dtype=self.dtype
            )
            self.tokenizer = MarianTokenizer.from_pretrained(TRANSLATION_MODEL_PATH)
        except:
            self.model = MarianMTModel.from_pretrained(
                self.model_id,
                device_map=self.device, 
                torch_dtype=self.dtype
            )
            self.tokenizer = MarianTokenizer.from_pretrained(self.model_id)

    def translate(self, text: str) -> str:
        lang, _ = classify(text)
        if lang == "en":
            return text

        tokens = self.tokenizer(
            text, 
            padding=self.padding, 
            truncation=self.truncation, 
            return_tensors="pt"
        ).to(self.device)
        translated_tokens = self.model.generate(**tokens)
        translated_text = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=self.skip_special_tokens)
        logger.info(f"translated product name: {translated_text}")

        return translated_text