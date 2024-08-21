import json
from pathlib import Path
import time
from typing import List, Tuple, Any

import torch
from torch import Tensor
from transformers import AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPast
from typing import Dict, List, Optional
from eval_utils import (
    dump_jsonl,
    create_prompt,
    load_data,
    get_answer,
    DATA_NAME_TO_MAX_NEW_TOKENS,
)
from vllm import LLM, SamplingParams
from args import parse_args


MAX_POSITION_ID = 128 * 1024  # Determined by the model
TRUNCATE_LEN = 128 * 1024

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1000)
def truncate_input(input: list, max_length: int, manner="middle"):
    if len(input) <= max_length:
        return input
    if manner == "middle":
        split = max_length // 2
        return input[0:split] + input[-split:]
    else:
        return None


def truncate_by_tokens(input, tok, max_tokens, manner: str = "middle"):
    tokens = tok.encode(input)
    len_before = len(tokens)
    print(f"# tokens before: {len_before}")
    tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
    len_after = len(tokens)  # type: ignore
    print(f"# tokens after: {len_after}")
    assert len_after <= len_before
    assert len_after <= max_tokens
    return tok.decode(tokens, skip_special_tokens=True)


def get_pred(
    model,
    tok: AutoTokenizer,
    input_text: str,
    max_tokens: int,
    verbose: bool = False,
) -> str:
    """
    Truncate down to 128k then make inference.
    """
    print("Truncating...")
    input_text = truncate_by_tokens(input_text, tok, TRUNCATE_LEN)
    if verbose:
        print("# chars:", len(input_text))
        print("=============== Input ===============")
        print(input_text[:200])
        print("...")
        print(input_text[-200:])
        print("=====================================")
    messages = [
    {"role": "user", "content": input_text},
    ]
    tokenized_chat = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    print(tok.decode(tokenized_chat[0])[:200])
    print(tok.decode(tokenized_chat[0])[-200:])
    # output = model(input_text)
    outputs = model.generate([input_text], sampling_params)
    # print(outputs)
    output = outputs[0].outputs[0].text
    print("Chunked generation:", output)
    return output

class HuggingFaceModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
        model_kwargs = {"attn_implementation": "flash_attention_2"}
        try:
            self.pipeline = pipeline(
                "text-generation",
                model=name_or_path,
                tokenizer=self.tokenizer,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                model_kwargs=model_kwargs,
            )
            print("pipeline")
        except:
            print("not using pipeline")
            self.pipeline = None
            self.model = AutoModelForCausalLM.from_pretrained(name_or_path, trust_remote_code=True,torch_dtype=torch.bfloat16,)
            
        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop')

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        if self.pipeline is None:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            output = self.model.generate(
                **inputs,
                **self.generation_kwargs
            )
            generated_text = self.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        else:
            print(self.generation_kwargs)
            output = self.pipeline(text_inputs=prompt, **self.generation_kwargs,)
            assert len(output) == 1
            generated_text = output[0]["generated_text"]
        # print(generated_text)
        # remove the input form the generated text
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt) :]
                
        if self.stop is not None:
            for s in self.stop:
                generated_text = generated_text.split(s)[0]
        # return {'text': [generated_text]}
        return generated_text

def load_model(
    model_name: str = "../../../yarn-mistral-7b-128k",
    ngpu=8,
):
    print("Loading tokenizer")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # tok.pad_token = tok.eos_token
    print("Loading model")
    start_time = time.time()
    llm = LLM(model=model_name, trust_remote_code=True)#, tensor_parallel_size=ngpu)
    # llm = HuggingFaceModel(
    # name_or_path=model_name,
    # do_sample=False,
    # repetition_penalty=1,
    # stop="",
    # max_new_tokens=1000,
    # )
    print("Time taken:", round(time.time() - start_time))
    return llm, tok  # type: ignore


if __name__ == "__main__":
    
    args = parse_args()
    model_name = args.model_name

    print(json.dumps(vars(args), indent=4))
    data_name = args.task

    # Model
    max_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]
    model, tok = load_model(args.model_path)
    # sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    # Data
    result_dir = Path(args.output_dir, model_name)
    result_dir.mkdir(exist_ok=True, parents=True)
    examples = load_data(data_name, data_dir=args.data_dir)

    if args.stop_idx is None:
        args.stop_idx = len(examples)
        output_path = (
            result_dir / f"preds_{data_name}.jsonl"
        )
    else:
        output_path = (
            result_dir / f"preds_{data_name}_{args.start_idx}-{args.stop_idx}.jsonl"  # noqa
        )

    preds = []
    print("==== Evaluation ====")
    print(f"# examples: {len(examples)}")
    print(f"Start index: {args.start_idx}")
    print(f"Stop index: {args.stop_idx}")
    print(f"Verbose: {args.verbose}")
    print(f"Max tokens: {max_tokens}")
    for i in range(args.start_idx, args.stop_idx):
        eg = examples[i]
        input_text = create_prompt(eg, data_name, model_name, args.data_dir)
        print(f"====== Example {i} ======")
        pred = get_pred(
            model, tok, input_text, max_tokens=max_tokens, verbose=args.verbose
        )
        if args.verbose:
            print(pred)
        preds.append(
            {
                "id": i,
                "prediction": pred,
                "ground_truth": get_answer(eg, data_name),
            }
        )
        dump_jsonl(preds, output_path)
