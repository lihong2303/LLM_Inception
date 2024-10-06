import random
import re
from data_loader import *
import logging
import re
import logging
import numpy as np
import argparse
import time
import base64
import requests
from io import BytesIO

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--type', type=str, required=True, help='attribute, affordance, action')
    parser.add_argument('--mode', type=str, required=True, help='nomem, strmem, nlmem, chainmem')
    parser.add_argument('--number', type=int, required=False, default=1, help='number of shot: 1 or 3')
    argss = parser.parse_args()

    return argss

attr_name = {'affordance': ['break', 'carry', 'clean', 'close', 'cut', 'eat', 'open','push','sit','write'],
             'attribute':  ['wooden', 'metal', 'flying', 'ripe', 'fresh', 'natural', 'cooked', 'painted', 'rusty', 'furry'],
             'action': ["hit", "push", "run", "dress", "drive", "cooking", "throw", "build", "shake", "cut"]}

api_key = "your api key here"

# Function to encode the image
def encode_image(image):
    byte_arr = BytesIO()
    image.save(byte_arr, format='PNG')
    image = byte_arr.getvalue()
    return base64.b64encode(image).decode('utf-8')


# Getting the base64 string
headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}


def get_payload(memory, query_img, img1, img2):
    payload = {
    "model": "gpt-4-turbo",
    "messages": [
        {
        "role": "user",
        "content": [
                {
                    "type": "text",
                    "text": f"Instruction: {memory} Determine the relationship between the original image and the candidate images, and select the images with the same {args.type} as the original image.\n",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/webp;base64,%s" % (encode_image(query_img)),
                    },
                },
                {
                    "type": "text",
                    "text": ".Candidate images: Image1: ",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/webp;base64,%s" % (encode_image(img1)),
                    },
                },
                {
                    "type": "text",
                    "text": ", Image2:",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/webp;base64,%s" % (encode_image(img2)),
                    },
                },
                {
                    "type": "text",
                    "text": ". Your response should be direct and exclusively only include one of the following items.\n Options: [Image1, Image2].",
                },
        ]
        }
    ],
    "max_tokens": 50
    }
    return payload


def get_attr_prompt(img1, img2):
    payload = {
    "model": "gpt-4-turbo",
    "messages": [
        {
        "role": "user",
        "content": [
                {
                    "type": "text",
                    "text": f"Give you two images, find there shared {args.type}: Image1:",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/webp;base64,%s" % (encode_image(img1)),
                    },
                },
                {
                    "type": "text",
                    "text": ", Image2:",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/webp;base64,%s" % (encode_image(img2)),
                    },
                },
                {
                    "type": "text",
                    "text": f".\n Your response should only include shared {args.type} in the following options.\n Options: {attr_name[args.type]}. Your response should only contain shared {args.type}, do not mention others!",
                },
        ]
        }
    ],
    "max_tokens": 100
    }
    return payload

    
def find_number_after_image(s):
    match = re.search(r'Image(\d+)', s)
    if match:
        return int(match.group(1))
    else:
        return None
    


instruct_question = ".Based on these knowledge, answer the following question:\n"

chainmem = []

def update_chainmem(answer, obj0, obj1):
    global chainmem
    attrs = []
    for attr in attr_name[args.type]:
        if attr in answer:
            attrs.append(attr)
    if args.type == 'action':
        chainmem.append([','.join(obj0),  ' '.join(attrs), ','.join(obj1)])
    else:
        chainmem.append([obj0, ' '.join(attrs), obj1])
    if len(chainmem) > 5:
        chainmem = chainmem[-5:]
        
        
memory = {}
given_weight = 1.0
forget_weight = 0.2
def update_memory(answer, obj):
    if args.mode == 'chainmem':
        update_chainmem(answer, obj[0], obj[1])
        return
    
    for attr in attr_name[args.type]:
        if attr in answer:
            if attr not in memory.keys():
                memory[attr] = {}
            for object_ in obj:
                if args.type == 'action':
                    object = object_[0]
                else:
                    object = object_
                if object not in memory[attr].keys():
                    memory[attr][object] = given_weight
                else:
                    memory[attr][object] += given_weight
    
    for attr, object in memory.items():
        for obj, weight in object.items():
            weight -= forget_weight
            if weight <= 0:
                memory[attr].pop(obj)
        if len(memory[attr]) == 0:
            memory.pop(attr)


def organize_memory(mode):
    if mode == 'strmem':
        structured_memory = {}
        for attr, object in memory.items():
            structured_memory[attr] = list(object.keys())
        return f"Given the memory{str(structured_memory)}, please answer the following question.\n"
    
    elif mode == 'nlmem':
        res = ""
        for attr, object in memory.items():
            objects = ""
            for obj in object:
                objects = objects + obj + ","
            if len(objects) > 0:
                objects = objects[:-1]
            res = res + objects + " has " + attr + f" {args.type}" + ".\n"
        return memory_base + res + instruct_question    
    
    elif mode == 'chainmem':
        res = ""
        for mem_idx, mem_item in enumerate(chainmem):
            if mem_idx == 0:
                res += "->".join(mem_item)
            else:
                res += "->"
                res += "->".join(mem_item[1:])       
        
        return memory_base + res + instruct_question

if __name__ == "__main__":

    global args
    global memory_base
    args = parse_arguments()

    memory_base = "Before this question, you have learnt that related pictures may have the following {args.type}:\n"

    if args.type != 'action':
        logging.basicConfig(filename=f"OCL_paired_{args.type}_{args.mode}_{args.number}.log", level=logging.INFO)
    else:
        logging.basicConfig(filename=f"Pangea_{args.type}_{args.mode}_{args.number}.log", level=logging.INFO)
        
    if args.type == "attribute":
        paired_options = ['furry-metal', 'fresh-cooked', 'natural-ripe', 'painted-rusty']
    elif args.type == "affordance":
        paired_options = ['sit-write', 'push-carry', 'cut-clean', 'open-break']
    elif args.type == "action":
        paired_options = ['run-51.3.2_hit-18.1', 'drive-11.5_dress-41.1.1-1-1',  'cooking-45.3_build-26.1' , 'shake-22.3-2_cut-21.1-1']
    
    for paired_attr in paired_options:
        
        logging.info(f"Paired_Attr: {paired_attr} BEGIN")
        step = []
        for epoch in range(50):  
            if args.type == "action":
                attr_bank = paired_attr.split('_')
                data = data_loader("./../data/Pangea", paired_attr, args.type)
            else:
                attr_bank = paired_attr.split('-')
                data = data_loader("./../data/OCL", paired_attr, args.type)
            memory = {}
            chainmem = []
            
            useless = 0 
        
            attr = random.choice(attr_bank)
            query_image, correct_image, false_image = data.get_data(attr)
            query_image_encoded = (data.process_img(query_image))
            correct_image_encoded = (data.process_img(correct_image))
            usage = 0
            if args.mode != "nomem":
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=get_attr_prompt(query_image_encoded, correct_image_encoded))
                if response.text:
                    response = response.json()
                else:
                    print(response)
                    i -= 1
                    continue
                if "usage" in response.keys():
                    prompt = response['choices'][0]['message']['content']
                    usage += response['usage']['total_tokens']
                    if args.type != "action":
                        update_memory(prompt, [data.data[query_image[0]]['objects'][query_image[1]]['obj'], data.data[correct_image[0]]['objects'][correct_image[1]]['obj']])
                    else:
                        update_memory(prompt, [data.data[query_image][2], data.data[correct_image][2]])
                
                if args.number == 3:
                    for _ in range(2):
                        attr = random.choice(attr_bank)
                        query_image, correct_image, false_image= data.get_data(attr)
                        query_image_encoded = (data.process_img(query_image))
                        correct_image_encoded = (data.process_img(correct_image))
                        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=get_attr_prompt(query_image_encoded, correct_image_encoded))
                        if response.text:
                            response = response.json()
                        else:
                            print(response)
                            i -= 1
                            continue
                        if "usage" in response.keys():
                            prompt = response['choices'][0]['message']['content']
                            usage += response['usage']['total_tokens']
                            if args.type != "action":
                                update_memory(prompt, [data.data[query_image[0]]['objects'][query_image[1]]['obj'], data.data[correct_image[0]]['objects'][correct_image[1]]['obj']])
                            else:
                                update_memory(prompt, [data.data[query_image][2], data.data[correct_image][2]])
                    
            for i in range(500):        
                if i % 5 == 4: 
                    # change to another attribute
                    if attr == attr_bank[0]:
                        attr = attr_bank[1]
                    else:
                        attr = attr_bank[0]
                query_image = correct_image
                correct_image, false_image = data.get_next(attr)
                    
                query_image_encoded = (data.process_img(query_image))
                correct_image_encoded = (data.process_img(correct_image))
                false_image_encoded = (data.process_img(false_image))
                dir = random.randint(0, 1)
                correct_ans = 0
                logging.info(f"Iteration{i}, memory:{organize_memory(args.mode)}")
                if dir:
                    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=get_payload(organize_memory(args.mode), query_image_encoded, correct_image_encoded, false_image_encoded))
                    correct_ans = 1
                else:
                    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=get_payload(organize_memory(args.mode), query_image_encoded, false_image_encoded, correct_image_encoded))
                    correct_ans = 2
                if response.text:
                    response = response.json()
                else:
                    print(response)
                    i -= 1
                    continue
                if "usage" in response.keys():
                    usage += response['usage']['total_tokens']
                else:
                    i -= 1
                    pass
                try:
                    answer = find_number_after_image(response['choices'][0]['message']['content'])
                except:
                    print(response)
                    continue
                if answer == None:
                    i -= 1
                    continue
                logging.info(f"Iteration{i}, gemini:{answer}, expected:{correct_ans}, attr:{attr}, Query:{query_image}, Correct:{correct_image}, False:{false_image}")
                if answer != correct_ans:
                    print(f"epoch{epoch+1}: length: {i - useless}")
                    step.append(i - useless)
                    break
                
                if args.mode != "nomem":
                    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=get_attr_prompt(query_image_encoded, correct_image_encoded))
                    if response.text:
                        response = response.json()
                    else:
                        print(response)
                        i -= 1
                        continue
                    if "usage" in response.keys():
                        prompt = response['choices'][0]['message']['content']
                        usage += response['usage']['total_tokens']
                        if args.type != "action":
                            update_memory(prompt, [data.data[query_image[0]]['objects'][query_image[1]]['obj'], data.data[correct_image[0]]['objects'][correct_image[1]]['obj']])
                        else:
                            update_memory(prompt, [data.data[query_image][2], data.data[correct_image][2]])
                
                logging.info(f"Iteration{i}, Overall usage{usage}")
        
        logging.info(f"mean: {np.mean(step)}, max: {np.max(step)}, {step}")
        logging.info(f"Paired_Attr: {paired_attr} END")
