import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:  # Optional dependency when using remote HTTP endpoints
    import requests  # type: ignore[import]
except ImportError:
    requests = None  # type: ignore[assignment]

try:  # Optional dependency when using OpenAI-compatible API
    from openai import OpenAI  # type: ignore[import]
except ImportError:
    OpenAI = None  # type: ignore[assignment]

import config_path
from Statistics import global_statistics


class AsyncQA_Ori:

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AsyncQA_Ori, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance
    
    def init_instance(self):
        if not self._initialized:
            self.qa_interface = setup_qa_ori()
            self.executor = ThreadPoolExecutor()
            self._initialized = True

    async def ask(self, question, system_msg=""):
        loop = asyncio.get_running_loop()

        result = await loop.run_in_executor(self.executor, self.qa_interface, question, system_msg)
        return result

    def close(self):
        self.executor.shutdown()

class AsyncQA_OpenFOAM_LLM:

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AsyncQA_OpenFOAM_LLM, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance
    
    def init_instance(self):
        if not self._initialized:
            self.qa_interface = setup_qa_openfoam_llm()
            self.executor = ThreadPoolExecutor()
            self._initialized = True

    async def ask(self, question, system_msg=""):
        loop = asyncio.get_running_loop()

        result = await loop.run_in_executor(self.executor, self.qa_interface, question, system_msg)
        return result

    def close(self):
        self.executor.shutdown()

def setup_qa_ori():

    local_path = getattr(config_path, "local_model_path", "") or ""
    if local_path and os.path.isdir(local_path):
        return setup_local_llm(local_path)

    if config_path.model and os.path.isdir(config_path.model):
        return setup_local_llm(config_path.model)

    def get_qwen_response(user_msg, system_msg=""):

        if OpenAI is None:
            raise RuntimeError("The 'openai' package is required for remote model access. Install it or configure 'local_model_path'.")

        client = OpenAI(api_key=os.environ.get("API_KEY"),
                        base_url=os.environ["BASE_URL"],
        )
        if system_msg == "":
            messages=[
                {
                    "role": "user",
                    "content": user_msg
                }
            ]
        else:
            messages=[
                {
                    "role": "system",
                    "content": system_msg
                },
                {
                    "role": "user",
                    "content": user_msg
                }
            ]
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=config_path.model,
            temperature=config_path.temperature
        )
        chat_completion_dict = dict(chat_completion)
        # print(chat_completion_dict.keys())
        usage = chat_completion_dict['usage']
        usage = dict(usage)
        total_tokens = usage['total_tokens']
        prompt_tokens = usage['prompt_tokens']
        completion_tokens = usage['completion_tokens']

        global_statistics.total_tokens += total_tokens
        global_statistics.prompt_tokens += prompt_tokens
        global_statistics.completion_tokens += completion_tokens

        return chat_completion.choices[0].message.content

    return get_qwen_response
    

def setup_local_llm(model_path):
    try:
        import torch  # type: ignore[import]
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError("Local model support requires 'transformers' and 'torch' to be installed") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = eos_token_id

    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        model.to("cpu")

    model.eval()

    max_new_tokens = getattr(config_path, "max_new_tokens", 1024) or 1024
    try:
        max_new_tokens = int(max_new_tokens)
    except (TypeError, ValueError):
        max_new_tokens = 1024

    temperature = getattr(config_path, "temperature", 0.0) or 0.0
    do_sample = temperature > 0

    def get_local_response(user_msg, system_msg=""):
        messages = []
        if system_msg:
            messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": user_msg})

        chat_kwargs = {"add_generation_prompt": True}

        if hasattr(tokenizer, "apply_chat_template"):
            prompt_inputs = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                **chat_kwargs,
            )
        else:
            parts = []
            if system_msg:
                parts.append(f"<|system|>\n{system_msg}\n")
            parts.append(f"<|user|>\n{user_msg}\n<|assistant|>\n")
            prompt_text = "".join(parts)
            prompt_inputs = tokenizer(prompt_text, return_tensors="pt")

        attention_mask = None
        if isinstance(prompt_inputs, dict):
            input_ids = prompt_inputs["input_ids"]
            attention_mask = prompt_inputs.get("attention_mask")
        elif hasattr(prompt_inputs, "input_ids"):
            input_ids = prompt_inputs.input_ids  # type: ignore[attr-defined]
            attention_mask = getattr(prompt_inputs, "attention_mask", None)
        else:
            input_ids = prompt_inputs

        if not hasattr(input_ids, "to"):
            input_ids = torch.tensor(input_ids)

        input_ids = input_ids.to(model.device)

        if attention_mask is not None:
            if not hasattr(attention_mask, "to"):
                attention_mask = torch.tensor(attention_mask)
            attention_mask = attention_mask.to(model.device)

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature if do_sample else None,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": eos_token_id,
        }
        generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
            )

        prompt_length = input_ids.shape[-1]
        generated_ids = output_ids[:, prompt_length:]
        response_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

        total_tokens = int(output_ids.shape[-1])
        prompt_tokens = int(prompt_length)
        completion_tokens = max(total_tokens - prompt_tokens, 0)

        global_statistics.total_tokens += total_tokens
        global_statistics.prompt_tokens += prompt_tokens
        global_statistics.completion_tokens += completion_tokens

        return response_text

    return get_local_response


def setup_qa_openfoam_llm():
    def get_openfoam_llm_response(user_msg, system_msg=""):
        if requests is None:
            raise RuntimeError("The 'requests' package is required for HTTP LLM access. Install it or configure 'local_model_path'.")
        base_url = config_path.openfoam_llm_base_url
        api_key =  os.getenv("API_KEY")
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }

        data = {
            'model': "qwen2.5-72b-instruct", 
            'messages': [],
            "max_tokens": 4000,
        }

        if system_msg:
            data['messages'].append({
                "role": "system",
                "content": system_msg
            })

        data['messages'].append({
            "role": "user",
            "content": user_msg
        })

        response = requests.post(base_url, headers=headers, json=data)
        if response.status_code == 200:
            chat_completion = response.json()

            usage = chat_completion['usage']
            usage = dict(usage)
            total_tokens = usage['total_tokens']
            prompt_tokens = usage['prompt_tokens']
            completion_tokens = usage['completion_tokens']
            global_statistics.total_tokens += total_tokens
            global_statistics.prompt_tokens += prompt_tokens
            global_statistics.completion_tokens += completion_tokens

            return chat_completion['choices'][0]['message']['content']
        
        else:
            print(f"Failed to get response: {response.status_code} {response.text}")
            return None

    return get_openfoam_llm_response