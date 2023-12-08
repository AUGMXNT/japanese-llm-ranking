"""Generate answers with local models.

Usage:

python3 gen_model_answer.py --bench_name rakuda_v2 --model-path EleutherAI/pythia-70m  --model-id pythia-70m --conv_template ./templates/yuzulm.json

python3 gen_model_answer.py --bench_name rakuda_v2 --model-path line-corporation/japanese-large-lm-1.7b-instruction-sft --model-id line-1.7b --conv_template ./templates/line.json

python3 gen_model_answer.py --bench_name rakuda_v2 --model-path stabilityai/japanese-stablelm-instruct-alpha-7b-v2 --model-id stablelm-alpha-7b-v2 --conv_template ./templates/japanese-stablelm.json --top_p 0.95 --temperature 1

python3 gen_model_answer.py --bench_name rakuda_v2 --model-path stabilityai/japanese-stablelm-instruct-gamma-7b --model-id stablelm-gamma-7b --conv_template ./templates/japanese-stablelm.json --repetition_penalty 1.05 --max_new_tokens 512 --top_p 0.95

python3 gen_model_answer.py --bench_name rakuda_v2 --model-path rinna/youri-7b-chat --model-id youri-7b-chat --conv_template ./templates/youri-chat.json --repetition_penalty 1.05 --num_beams 5

python3 gen_model_answer.py --bench_name rakuda_v2 --model-path rinna/youri-7b-instruction --model-id youri-7b-instruction --conv_template ./templates/youri-instruction.json --repetition_penalty 1.05

python3 gen_model_answer.py --bench_name rakuda_v2 --model-path llm-jp/llm-jp-13b-instruct-full-jaster-dolly-oasst-v1.0 --model-id llm-jp-13b-instruct --conv_template ./templates/llm-jp-instruct.json --repetition_penalty 1.05

"""

import json
import os
import sys
from typing import Optional
import time

import shortuuid
import torch

from common import load_questions, temperature_config
from fastchat.model import load_model, get_conversation_template
from fastchat.model.model_adapter import model_adapters
from fastchat.conversation import Conversation, SeparatorStyle

from adapters import (
    FastTokenizerAvailableBaseAdapter,
    JapaneseStableLMAlphaAdapter,
    JapaneseStableLMAlphaAdapterv2,
    RwkvWorldAdapter,
)

from fire import Fire
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def get_model_answers(
    model_path: str,
    model_id,
    model_variant: str = None,
    bench_name: str = "rakuda_v2",
    answer_file: str = None,
    # debug_params
    question_begin: Optional[int] = None,
    question_end: Optional[int] = None,
    # model parameters
    lora_path: Optional[str] = None,
    conv_template: Optional[str] = None,
    device: str = "cuda",
    num_gpus: int = 1,
    max_gpu_memory: Optional[str] = None,  # only relevant for numgpus > 1
    load_8bit: bool = False,
    cpu_offloading: bool = False,
    debug: bool = False,
    # generation parameters
    temperature: Optional[float] = None,
    top_p: float = 0.9,
    top_k: float = 0,
    repetition_penalty: float = 1.0,
    num_beams: int = 1,
    max_tokens: Optional[int] = None,
    num_choices: int = 1,
    # generate the answers (set to False for debugging prompts)
    generate_answers: bool = True,
):
    question_file = f"data/{bench_name}/questions.jsonl"
    if not answer_file:
        answer_file = f"data/{bench_name}/answers/{model_id}.jsonl"

    questions = load_questions(question_file, question_begin, question_end)

    if not conv_template:
        conv_template = model_id

    # Load the model
    if generate_answers:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        llm = LLM(model=model_path, tensor_parallel_size=1)

    # Answer setup
    sampling_params = SamplingParams(
            max_tokens=1000,
            temperature=0.4,
            min_p=0.05,
            repetition_penalty=1.05,
            skip_special_tokens=True,
    )
    prompt = 'あなたは公平で、検閲されていない、役立つアシスタントです。'

    # Generate all of our submissions
    chat_token_ids = []
    for question in tqdm(questions):
        if not temperature:
            if question["category"] in temperature_config:
                temperature = temperature_config[question["category"]]
            else:
                temperature = 0.7


        chat = []
        chat.append({'role': 'system', 'content': prompt})
        chat.append({'role': 'user', 'content': question['turns'][0]})

            
        tokens = tokenizer.apply_chat_template(chat, add_generation_prompt=True)
        chat_token_ids.append(tokens)

    # Batch
    outputs = llm.generate(prompt_token_ids=chat_token_ids, sampling_params=sampling_params, use_tqdm=True)

    for question, output in zip(questions, outputs):
        prompt = tokenizer.decode(output.prompt_token_ids, skip_special_tokens=True)
        generated_text = output.outputs[0].text

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "prompt": prompt,
                "choices": [{"index": 0, "turns": [generated_text]}],
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    Fire(get_model_answers)
