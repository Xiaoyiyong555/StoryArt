import argparse
import requests
import json
from PIL import Image
from io import BytesIO
import base64
import os

import re
from tqdm import tqdm
import glob
import torch
import accelerate
import random
from accelerate import PartialState
from collections import Counter
from storyart.novel2img import Novel2Img
import storyart.utils as utils


def parse_line_debug(line):
    line = line.strip()
    if "，	" in line:
        line_id, sentence = line.split("，	")[0], line.split("，	")[1]
        sentence = re.sub(r'【.*?】', '', sentence)
        return line_id, sentence
    else:
        return 0, None
    
def parse_line(line, chapter_id):
    line_info = {}
    
    if "|" in line:
        line_id, sentence = line.split("|")[0].strip(), line.split("|")[1].strip()
        line_info["line_id"] = line_id
        line_info["sentence"] = sentence
        line_info["chapter_id"] = chapter_id
    else:
        line_info["line_id"] = 0
        line_info["sentence"] = None
        line_info["chapter_id"] = chapter_id

    return line_info
    
def open_novel_file(file_list: list, basename=None) -> list:

    contents = []
    
    for file_info in file_list:
        file_path, chapter_id = file_info["file_path"], file_info["chapter_id"]
        if file_path.startswith("http"):
            file_name = file_path.split("/")[-1]
            r = requests.get(file_path)
            if r.status_code == 200:
                with open(file_name, 'wb') as f:
                    f.write(r.content)
                print("文件已下载：", file_name)
                file_path = file_name
            else:
                print("下载失败！")
                return
        try:
            basename = re.match(r'^([\u4e00-\u9fa5]+)', os.path.basename(file_path)).group(1)[:-1] if not basename else basename
            with open(file_path, 'r') as f:
                content = f.readlines()
            contents += list(map(lambda x: parse_line(line=x, chapter_id=chapter_id), content))
        except FileNotFoundError:
            print("文件不存在！")

    contents = list(filter(lambda x: x["sentence"]!=None, contents))
    contents = list(sorted(contents, key=lambda x: int(x["line_id"].split("-")[0])*10000 + int(x["line_id"].split("-")[1])))
    return contents, basename


def commbine_info(character_sets, info_list):
    counts = Counter([info["character"] for info in info_list if info["character"]!= "无"])
    name_counts = list(sorted(counts.items(), key=lambda x: -x[1]))

    character_list = []
    for info in info_list:
        if info["character"] not in character_sets and info["character"] != "无":
            character_sets[info["character"]] = ""
            character_list.append((info["character"], info["history"]))
            
    return character_sets, character_list, name_counts
        
        
def ip_insert(character_sets, character_results, name_counts, ratio=0.3, min_count=10):

    pos_prompt = "solo,masterpiece,best quality,very aesthetic,absurdres"
    with open("configs/character.txt", "r") as f:
        lines = f.readlines()
    f.close()
    ip_pools = {"male":set(), "female":set()}
    for line in lines:
        info = line.strip().split(", ")
        if info[0] == "1girl":
            ip_pools["female"].add(",".join(info[1:]))
        else:
            ip_pools["male"].add(",".join(info[1:]))

    character_results = dict(character_results)
    male_his = ip_pools["male"].copy()
    female_his = ip_pools["female"].copy()
    for i, name_count in enumerate(name_counts):
        name, count = name_count
        if i<(len(name_counts)*ratio) or count >= min_count or i<len(male_his):
            character_result = character_results[name]
            sex = character_result.split(",")[1]
            if sex in ["young man", "middel age man", "old man"]:
                if len(ip_pools["male"]) == 0:
                    ip_pools["male"] = male_his.copy()
                ip = [random.choice(list(ip_pools["male"]))]
                ip_pools["male"].remove(ip[0])
    
            elif sex in ["young women", "middel age woman", "old woman"]:
                if len(ip_pools["female"]) == 0:
                    ip_pools["female"] = female_his.copy()
                ip = [random.choice(list(ip_pools["female"]))]
                ip_pools["female"].remove(ip[0])
            else:
                ip = []
            
            ip_result = ",".join(character_result.split(",")[:2] + ip + character_result.split(",")[2:] + [pos_prompt])
            character_sets[name] = ip_result
        else:
            character_sets[name] = character_results[name]
    
    return character_sets


def pad_info(info_list):

    def find_near_info(index, key):
        hi = index
        info = "无"
        while hi>0 and info=="无":
            info = info_list[hi][key]
            hi -= 1
        if info!= "无":
            return info
        hi = index
        while hi<len(info_list) and info=="无":
            info = info_list[hi][key]
            hi += 1
        return info


    for i in range(len(info_list)):
        for key in ["scene"]:
            if info_list[i][key] == "无":
                info_list[i][key] = find_near_info(index=i, key=key)
                
    return info_list

def parse_args(input_args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--img_model_path", type=str, default="/data1/yiyong/checkpoints/animagine-xl-3.1")
    parser.add_argument("--input_path", type=str, default="novel")
    parser.add_argument("--save_path", type=str, default="/data3/yiyong/experiments/Novel-Cartoon")
    parser.add_argument("--context", type=str, default="现代")
    parser.add_argument("--history_len", type=int, default=30)


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
        
    return args

if __name__ == "__main__":
    
    args = parse_args()
    
    img_model_path = args.img_model_path
    history_len = args.history_len
    save_path = args.save_path
    context = args.context


    distributed_state = PartialState()
    
    pipe = Novel2Img(img_path=img_model_path, 
                     img_device=distributed_state.device, text_device=distributed_state.device, lowvarm=False)
    
    character_sets = {}
    file_path = glob.glob(f'{args.input_path}/*.txt')

    file_list = list(map(lambda x: {"chapter_id":int(os.path.basename(x).split(".")[0]), "file_path": x}, file_path))
    raw_lines, basename = open_novel_file(file_list, basename="test")

    lines = [] # load history
    history = []
    for i, raw_line in enumerate(raw_lines):
        if len(history) < history_len:
            history += [raw_line["sentence"]]
        else:
            history = history[1:] + [raw_line["sentence"]]
        line = raw_line.copy()
        line["history"] = history
        lines.append(line)

    utils.load_model(pipe.text_model ,distributed_state.device)
    # 0. info extract
    info_path = f"{save_path}/{basename}/info"
    os.makedirs(info_path, exist_ok=True)
    info_list = []
    
    inputs_lines = list(filter(lambda x: not os.path.exists(f'{info_path}/{x["line_id"]}.json'), lines))

    with distributed_state.split_between_processes(inputs_lines) as partial_lines:
        for line in tqdm(partial_lines):

            info = pipe.info_extract(line=line["sentence"], novel=line["history"])
            result = {"line_id":line["line_id"], "character":info["character"],
                    "scene":info["scene"], "sentence":line["sentence"]}
            
            with open(f'{info_path}/{line["line_id"]}.json', "w") as f:
                f.write(json.dumps(result, ensure_ascii=False))
            f.close()

    distributed_state.wait_for_everyone()
    p = os.listdir(info_path)
    for fn in p:
        with open(f"{info_path}/{fn}", "r") as f:
            info_list.append(json.loads(f.readlines()[0].strip()))
        f.close()
    info_list = list(sorted(info_list, key=lambda x: int(x["line_id"].split("-")[0])*10000 + int(x["line_id"].split("-")[1])))
    info_list = pad_info(info_list=info_list)

    for i in range(len(info_list)):
        info_list[i]["history"] = lines[i]["history"]

    if distributed_state.on_main_process():
        print("finish info extract")
    

    # 1. extract character
    
    character_sets, character_list, name_counts = commbine_info(character_sets, info_list)

    # 2. character set
    character_path = f"{save_path}/{basename}/character_setting"
    os.makedirs(character_path, exist_ok=True)
    
    partial_character_results = []
    character_results = []
    if not os.path.exists(f"{character_path}/{distributed_state.process_index}.txt"):
        with distributed_state.split_between_processes(character_list) as partial_character_list:
            for character in tqdm(partial_character_list):
                character_set = pipe.character_setting(character=character[0], novel=character[1])
                partial_character_results.append(f'{character[0]}||{character_set}')

        with open(f"{character_path}/{distributed_state.process_index}.txt","w") as f:
            f.write("\n".join(partial_character_results))
        f.close()

    distributed_state.wait_for_everyone()

    if distributed_state.process_index == 0:
         if not os.path.exists(f"{save_path}/{basename}/character_sets.json"):
            p = os.listdir(character_path)
            for fn in p:
                with open(f"{character_path}/{fn}", "r") as f:
                    character_results += list(map(lambda x: x.strip().split("||"), f.readlines()))
                f.close()
            character_sets = ip_insert(character_sets, character_results, name_counts)
            with open(f"{save_path}/{basename}/character_sets.json","w") as f:
                json.dump(character_sets, f, ensure_ascii=False, indent=2)
            f.close()

    distributed_state.wait_for_everyone()

    if distributed_state.on_main_process():
        print("finish character setting")


    # 3. set character
    with open(f"{save_path}/{basename}/character_sets.json","r") as f:
        character_sets = json.load(f)
    f.close()
    for name in character_sets:
        pipe.set_character(name, character_sets[name])

    # 4. novel to describe
    describe_path = f"{save_path}/{basename}/describe"
    os.makedirs(describe_path, exist_ok=True)
    describe_list = []

    for i in range(len(lines)):
        info_list[i]["chapter_id"] = lines[i]["chapter_id"]
        info_list[i]["history"] = lines[i]["history"]
    
    
    input_info_list = list(filter(lambda x: not os.path.exists("{}/{}/{}.txt".format(describe_path, x["chapter_id"], x["line_id"])), info_list[:10000]))
    with distributed_state.split_between_processes(input_info_list) as partial_info_list:
        for info in tqdm(partial_info_list):
                
            describe = pipe.word2describe(info={"character":info["character"],"scene": info["scene"]},
                                        line=info["sentence"], context=context, novel=info["history"])
            
            os.makedirs("{}/{}".format(describe_path, info["chapter_id"]), exist_ok=True)
            with open("{}/{}/{}.txt".format(describe_path, info["chapter_id"], info["line_id"]), "w") as f:
                f.write("\n".join([info["sentence"], describe, json.dumps({"character":info["character"], "scene": info["scene"]}, ensure_ascii=False)]))
            f.close()

    distributed_state.wait_for_everyone()
    
    for chapter_id in os.listdir(describe_path):
        for fn in os.listdir(f'{describe_path}/{chapter_id}'):
            with open(f"{describe_path}/{chapter_id}/{fn}", "r") as f:
                describe = f.readlines()[1].strip()
                describe_list.append({"chapter_id":chapter_id, "line_id":fn[:-4], "describe":describe})
            f.close()

    describe_list = list(sorted(describe_list, key=lambda x: int(x["line_id"].split("-")[0])*10000 + int(x["line_id"].split("-")[1])))
    
    if distributed_state.on_main_process():
        print("finish word2describe")
    
    utils.unload_model(pipe.text_model)

    utils.load_model(pipe.img_model ,distributed_state.device)
    pipe.img_model.enable_xformers_memory_efficient_attention()
    # 5. describe to image
    image_path = f"{save_path}/{basename}/images"
    os.makedirs(image_path, exist_ok=True)
    
    
    input_describe_list = list(filter(lambda x: not os.path.exists("{}/{}/{}_0.jpg".format(image_path, x["chapter_id"], x["line_id"])), describe_list[:5000]))
    with distributed_state.split_between_processes(input_describe_list) as partial_describe_list:
        for describe in tqdm(partial_describe_list):
        
            images = pipe.describe2imgs(describe=describe["describe"], batch_size=1, style="test",
                                        context=context, height=1216, width=832, steps=20)

            for i, np_image in enumerate(images):
                os.makedirs("{}/{}".format(image_path, describe["chapter_id"]), exist_ok=True)
                Image.fromarray(np_image).save("{}/{}/{}_{}.jpg".format(image_path, describe["chapter_id"], describe["line_id"], i))
    
    utils.unload_model(pipe.img_model)
    distributed_state.wait_for_everyone()

    if distributed_state.on_main_process():
        print("finish describe2image")
            
             

