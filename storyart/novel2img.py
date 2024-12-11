import os
import numpy as np
import torch
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, FluxPipeline
from omegaconf import OmegaConf

import random
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import json
from .utils import print_logger, unload_model, load_model, chat
from nltk.metrics import edit_distance


def parse_prompts_yaml(file_path):
    prompt_dict = OmegaConf.load(file_path)
    prompt_dict = json.loads(json.dumps(OmegaConf.to_container(prompt_dict)))["prompt"]

    for key in prompt_dict:
        prompt_dict[key] = [prompt_dict[key]["user"], prompt_dict[key]['anw'] if prompt_dict[key]['anw'] else ""] 
    return prompt_dict

def truncate_text(text, max_length, min_length):
    if len(text) <= max_length:
        return text

    last_comma_index = text.rfind(',', 0, max_length)
    if last_comma_index != -1 and len(text[:last_comma_index])>min_length:
        return text[:last_comma_index]

    return text[:max_length]


class Novel2Img():
    def __init__(self, img_path=None, text_path=None, config_path=".",
                 img_device="cuda:0", text_device="cuda:0", lowvarm=True):
        
        self.lowvarm = lowvarm
        self.config_path = config_path
        print_logger("start init")
        self.img_device = img_device
        self.img_model = FluxPipeline.from_pretrained(
            img_path,
            # variant="fp16",
            torch_dtype=torch.float16
        ).to(img_device)
        # self.img_model.scheduler = EulerAncestralDiscreteScheduler.from_config(self.img_model.scheduler.config)


        if self.lowvarm:
            unload_model(self.img_model)
            print_logger("LOWVARM MODE unload img_model to cpu")

        print_logger("finished load img_model")


        self.text_device = text_device
        
        self.tokenizer = AutoTokenizer.from_pretrained(text_path, trust_remote_code=True)
        self.text_model = AutoModelForCausalLM.from_pretrained(
            text_path,
            device_map="cpu",
            trust_remote_code=True
        ).eval().to(text_device)
        self.text_model.generation_config = GenerationConfig.from_pretrained(text_path, trust_remote_code=True)

        
        print_logger("finished load text_model")

        if self.lowvarm:
            self.text_device = self.img_device
            unload_model(self.text_model)

        # load style
        self.set_style()

         # load context
        self.set_context()

        # load prompt config
        self.prompt_templte = parse_prompts_yaml(f"{self.config_path}/prompt.yaml")

        self.name_to_sex = {}

        self.scene_describe, self.character_describe = "", ""
        self.character = "无"

    def set_context(self):
        # load context
        with open(f"{self.config_path}/context.json", "r") as f:
            templte = json.load(f)
        f.close()

        self.context_templte = {}
        for item in templte:
            name = item["name"]
            prompt = item["prompt"]
            negative_prompt = item["negative_prompt"]
            self.context_templte[name] = {"prompt": prompt, "negative_prompt": negative_prompt}

        # load ip
        with open(f"{self.config_path}/character.txt", "r") as f:
            lines = f.readlines()
        f.close()
        self.ip_pools = {"male":set(), "female":set()}
        for line in lines:
            info = line.strip().split(", ")
            if info[0] == "1girl":
                self.ip_pools["female"].add(",".join(info[1:]))
            else:
                self.ip_pools["male"].add(",".join(info[1:]))

        self.ip_pools["male"] = list(self.ip_pools["male"])
        self.ip_pools["female"] = list(self.ip_pools["female"])
        random.shuffle(self.ip_pools["male"])
        random.shuffle(self.ip_pools["female"])

        self.ip_idx = [0, 0] # male, female

    def set_style(self):
        # load style
        with open(f"{self.config_path}/style.json", "r") as f:
            templte = json.load(f)
        f.close()
        
        self.style_templte = {}
        for item in templte:
            name = item["name"]
            prompt = item["prompt"]
            negative_prompt = item["negative_prompt"]
            self.style_templte[name] = {"prompt": prompt, "negative_prompt": negative_prompt}


    def set_character(self, name, sex):
        if len(name)>=1:
            self.name_to_sex[name] = sex
            print_logger(f"sucess setting {name} -> {sex}")
        else:
            print_logger(f"unsucess setting name: {name}")


    def ip_insert(self, character_describe):

        pos_prompt = "solo,masterpiece,best quality,very aesthetic,absurdres"
        sex = character_describe.split(",")[1]

        if sex in ["young man", "middel age man", "old man"]:
            ip = self.ip_pools["male"][self.ip_idx[0]]
            self.ip_idx[0] = (self.ip_idx[0] + 1)%len(self.ip_pools["male"])
            
        elif sex in ["young women", "middel age woman", "old woman"]:
            ip = self.ip_pools["female"][self.ip_idx[1]]
            self.ip_idx[1] = (self.ip_idx[1] + 1)%len(self.ip_pools["female"])
        else:
            ip = ""

        ip_describe = ",".join(character_describe.split(",")[:2] + [ip] + character_describe.split(",")[2:] + [pos_prompt])
        
        return ip_describe

    def character_setting(self, character, novel) -> str:

        character_prompt = self.prompt_templte["character_prompt"][0]
        history = [(self.prompt_templte["read_novel"][0].format(novel=novel), self.prompt_templte["read_novel"][1])]
        query = character_prompt.format(name=character)
        response, _ = chat(self.text_model, self.tokenizer, query, history=history)

        print_logger("character setting: " + response)
        try:
            s, e = response.find("{"), response.find("}")
            character_json = json.loads(response[s:e+1])
            sex, hair, clothes = character_json["sex"], character_json["hair"], character_json["clothes"]
            describe = f"人,{sex},{hair},{clothes}"
            
        except:
            print_logger("character set error!")
            describe = "默认,young man,black short hair,black t-shirt"
        
        describe = self.ip_insert(describe)
        
        return describe
    
    def info_extract(self, novel, line) -> json:

        self.text_model.generation_config.top_p = 0.8
        self.text_model.generation_config.temperature = 0.95

        if self.lowvarm:
            load_model(self.text_model,self.text_device)

        history = [(self.prompt_templte["read_novel"][0].format(novel=novel), self.prompt_templte["read_novel"][1])]
        query = self.prompt_templte["info_extract"][0].format(line=line)
        
        response, _ = chat(self.text_model, self.tokenizer, query, history=history)
        s = response.find("{")
        e = response.find("}")+1

        print_logger("info extract: " + response)
        try:
            response = json.loads(response[s:e])
            response =  {"character": response["character"], "scene": response["scene"]}
        except:
            print_logger("info extract error!")
            response = {"character":"无", "scene":"无"}
        torch.cuda.empty_cache()
        
        if self.lowvarm:
            unload_model(self.text_model)

        return response
    
    def word2describe(self, info, line, context, novel) -> str:

        self.text_model.generation_config.top_p = 1.0
        self.text_model.generation_config.temperature = 0.7
        if self.lowvarm:
            load_model(self.text_model,self.text_device)

        #0. character
        character = None
        
        character_describe = ""
        if info["character"] != "无" and len(info["character"])>1:
            character = info["character"]
            # if note exist character set
            if sum([character in n for n in self.name_to_sex]) == 0 and sum([n in character for n in self.name_to_sex]) == 0:
                
                describe = self.character_setting(character, novel)
                
                self.set_character(name=character, sex=describe)

            # search character describe
            min_distance = float("inf")
            for name in self.name_to_sex:
                if name in character or character in name:
                        distance = edit_distance(name, character)
                        if distance < min_distance:
                            min_distance = distance
                            character_describe = ",".join(self.name_to_sex[name].split(",")[1:])


        self.character_describe = character_describe
        self.character = character

        #1. scene
        scene_prompt = self.prompt_templte["scene_prompt"][0]
        scene_describe = None
        if info["scene"] != "无":
            scene = info["scene"]
            
            response, _ = chat(self.text_model, self.tokenizer, scene_prompt.format(scene=scene, style=context), history=[])
            
            print_logger("scene setting: " + response)
            try:
                scene_describe = json.loads(response)['image_prompt']
                scene_describe =truncate_text(scene_describe, max_length=70, min_length=20)

            except:
                print_logger("error scene predict")
     
        scene_describe = self.scene_describe if not scene_describe else scene_describe
        
        self.scene_describe = scene_describe

        #2. pose
        pose_prompt = self.prompt_templte["pose_prompt"][0]
        pose_describe = ""
        if character:
            response, _ = chat(self.text_model, self.tokenizer, pose_prompt.format(character=character, line=line), history=[])
            print_logger(f"pose predict: {response}")
            try:
                pose_describe = json.loads(response)['answer']
            except:
                print_logger("pose predict error!")

        if self.lowvarm:
            unload_model(self.text_model)

        return f'{character_describe}, {pose_describe}, {scene_describe}'
    

    
    def describe2imgs(self, describe="", batch_size=1, style="base_coun", context="现代",
                    height=1024, width=1024, steps=30) -> list[np.array]:
        
            
        if self.lowvarm:
            load_model(self.img_model, self.img_device)
        
        # 0. online prompt templte load
        self.set_style()
        self.set_context()
        prompt = self.style_templte[style]["prompt"].format(prompt=describe)
        if len(self.context_templte[context]["prompt"])>3:
            prompt = prompt + "," + self.context_templte[context]["prompt"]
        negative_prompt = self.style_templte[style]["negative_prompt"]
        if len(self.context_templte[context]["negative_prompt"])>3:
            negative_prompt = self.context_templte[context]["negative_prompt"] + ", " + negative_prompt

        # 1. describe to images
        imgs = self.img_model(prompt=prompt,
                            #    negative_prompt=negative_prompt,
                                num_inference_steps=steps, num_images_per_prompt=batch_size,guidance_scale=3.5,
                                height=height, width=width).images
        imgs = [np.array(img,dtype=np.uint8) for img in imgs]
        
        if self.lowvarm:
            unload_model(self.img_model)


        return  imgs
        



        
        