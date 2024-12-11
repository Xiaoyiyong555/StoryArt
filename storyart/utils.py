from __future__ import annotations

import pynvml
import time
import torch
import datetime


def print_logger(message):
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\033[92m[{timestamp}] {message}\033[0m")
    
def load_model(model, device):
    model.to(device)

def unload_model(model):
    model.to("cpu")
    torch.cuda.empty_cache()

def check_gpu_memory(memory):
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    
    while True:
        for i in range(0, device_count):
            
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_memory = info.free / 1024 / 1024 / 1024 
            if free_memory > memory and len(processes) < 2:
                pynvml.nvmlShutdown()
                # return f"cuda:{str(i)}"
                return i 
        time.sleep(100)

def chat(model, tokenizer, query, history):

    conversation = []
    
    for query_h, response_h in history:
        conversation.append({"role": "user", "content": query_h})
        conversation.append({"role": "assistant", "content": response_h})
    conversation.append({"role": "user", "content": query})
    input_text = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    model_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    history.append((query, response))

    return response, history