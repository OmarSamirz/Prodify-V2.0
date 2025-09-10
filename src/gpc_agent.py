from teradataml import *
from langgraph.graph import StateGraph, START, END
from transformers import AutoModelForCausalLM, AutoTokenizer

import re
import time
import json
from dataclasses import dataclass
from typing import Dict, List, TYPE_CHECKING

from modules.logger import logger
from modules.db import TeradataDatabase
from utils import load_translation_model
from constants import OPUS_TRANSLATION_CONFIG_PATH, DEVICE

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams


@dataclass
class GpcState(Dict):
    product_description: str
    segment: str = ""
    family: str = ""
    class_: str = ""
    brick: str = ""


class GpcAgent:

    def __init__(
        self,
        model_name: str = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit",
        use_translation: bool = False,
        use_vllm: bool = False,
    ) -> None:
        self.model_name = model_name
        self.use_translation = use_translation
        self.use_vllm = use_vllm
        self.tokenizer = None
        self.model = None
        self.td_db = None
        self.df = None
        self.agent = None
        self.translation_model = None
        
        self._setup_model()
        self._setup_database()
        self._setup_translation()
        self._build_graph()

    def _setup_model(self) -> None:
        if self.use_vllm:
            self.model = LLM(
                model=self.model_name,
                dtype="bfloat16",
                max_num_seqs=8,
                gpu_memory_utilization=0.8,
                max_model_len=10240,
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map=DEVICE,
            )
    
    def _setup_database(self) -> None:
        self.td_db = TeradataDatabase()
        self.td_db.connect()
        self.df = DataFrame.from_table("gpc").to_pandas()

    def _setup_translation(self) -> None:
        if self.use_translation:
            self.translation_model = load_translation_model(OPUS_TRANSLATION_CONFIG_PATH)

    def _get_few_shot_examples(self) -> str:
        return """
        Example 1:
        Input:
        {
            "product_description": "Chicken Breasts (0.5Kg)",
            "segment": "Food/Beverage",
            "family": "Meat/Poultry/Other Animals",
            "class_": "Meat/Poultry/Other Animals - Unprepared/Unprocessed",
            "brick": "Alternative Meat/Poultry/Other Animal Species - Unprepared/Unprocessed"
        }

        Example 2:
        Input:
        {
            "product_description": "Indian Menshawi Mango (0.5Kg)",
            "segment": "Food/Beverage",
            "family": "Fruits - Unprepared/Unprocessed (Fresh)",
            "class_": "Fruits - Unprepared/Unprocessed (Fresh) Variety Packs",
            "brick": "Fruits - Unprepared/Unprocessed (Fresh) Variety Packs"
        }

        Example 3:
        Input:
        {
            "product_description": "bonomi cocoa butter biscuits - 150 g",
            "segment": "Food/Beverage",
            "family": "Bread/Bakery Products",
            "class_": "Biscuits/Cookies",
            "brick": "Biscuits/Cookies (Shelf Stable)"
        }
        """
    
    def _segment_node(self, state: GpcState) -> GpcState:
        candidates = self.df["SegmentTitle"].drop_duplicates().tolist()
        state["segment"] = self._classify_node(state["product_description"], candidates)
        return state
    
    def _family_node(self, state: GpcState) -> GpcState:
        candidates = self.df[self.df["SegmentTitle"] == state["segment"]]["FamilyTitle"].drop_duplicates().tolist()
        state["family"] = self._classify_node(state["product_description"], candidates)
        return state

    def _class_node(self, state: GpcState) -> GpcState:
        candidates = self.df[self.df["FamilyTitle"] == state["family"]]["ClassTitle"].drop_duplicates().tolist()
        state["class_"] = self._classify_node(state["product_description"], candidates)
        return state

    def _brick_node(self, state: GpcState) -> GpcState:
        candidates = self.df[self.df["ClassTitle"] == state["class_"]]["BrickTitle"].drop_duplicates().tolist()
        state["brick"] = self._classify_node(state["product_description"], candidates)
        return state
    
    def _classify_node(self, text: str, candidates: List[str]) -> str:
        if len(candidates) == 1:
            return candidates[0]

        few_shot_examples = self._get_few_shot_examples()
        prompt = (
        "You are an expert, unbiased product classification AI designed to assign the most accurate product category based on the given product name and a list of possible categories.\n"
        "Follow these steps for every query:\n"
        "1. Analyze the product name for descriptive keywords and context clues.\n"
        "2. Determine its primary purpose and intended use.\n"
        "3. Compare the product to the categories, explaining your semantic reasoning.\n"
        "4. If uncertain, pick the most specific or least ambiguous fit and state your uncertainty via the confidence score.\n"
        "5. Respond only in the following JSON format.\n"
        f"{few_shot_examples}\n"
        f"CLASSIFY THIS PRODUCT:\n"
        f"Product Name: \"{text}\"\n"
        "Categories:\n" + "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidates)]) +
        "\n\nRespond ONLY in strict JSON format:\n"
        "{\n"
        "  \"category_number\": <number>,\n"
        "  \"category_name\": \"<name>\",\n"
        "  \"confidence\": <0.0-1.0>,\n"
        "}\n"
        )
        if self.use_vllm:
            sampling_params = SamplingParams(temperature=0.0, max_tokens=100)
            outputs = self.model.generate([prompt], sampling_params)
            output_text = outputs[0].outputs[0].text.strip()
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device) 
            output_ids = self.model.generate(**inputs, max_new_tokens=100)[0][len(inputs.input_ids[0]):] 
            output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        try:
            match = re.search(r'(\{.*?\})', output_text)
            if match:
                result = match.group(1).strip()
                data = json.loads(result)
                if data["category_name"] in candidates:
                    return data["category_name"]
                if "category_number" in data:
                    return candidates[int(data["category_number"])]
        except json.JSONDecodeError:
            output_text_lower = output_text.lower()
            for c in candidates:
                if c.lower() in output_text_lower:
                    return c

            logger.info(f"The model did not choose from the list output text: {output_text}.")
            return candidates[0]
    
    def _build_graph(self):
        workflow = StateGraph(GpcState)
        workflow.add_node("segment", self._segment_node)
        workflow.add_node("family", self._family_node)
        workflow.add_node("class", self._class_node)
        workflow.add_node("brick", self._brick_node)

        workflow.add_edge(START, "segment")
        workflow.add_edge("segment", "family")
        workflow.add_edge("family", "class")
        workflow.add_edge("class", "brick")
        workflow.add_edge("brick", END)

        self.agent = workflow.compile()

    def classify(self, product_name: str) -> Dict:
        start_time = time.time()
        
        processed_name = product_name
        if self.translation_model is not None:
            processed_name = self.translation_model.translate(product_name).lower()
        
        result = self.agent.invoke({"product_description": processed_name})
        
        end_time = time.time()
        return {
            "original_name": product_name,
            "processed_name": processed_name,
            "segment": result["segment"],
            "family": result["family"],
            "class": result["class_"],
            "brick": result["brick"],
            "duration_seconds": end_time - start_time,
        }