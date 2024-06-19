import base64
import requests
import json
import random
import re
import sys
from data_loader_two import *
from io import BytesIO
import logging
import numpy as np
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--mode', type=str, required=True, help='nomem, strmem, nlmem')
    parser.add_argument('--number', type=int, required=False, default=1, help='number of shot: 1 or 3')
    args = parser.parse_args()

    return args

# OpenAI API Key
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
                    "text": f"Instruction: {memory} Determine the relationship between the original image and the candidate images, and select the images with the same attributes as the original image.\n",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/webp;base64,%s" % (query_img),
                    },
                },
                {
                    "type": "text",
                    "text": ".Candidate images: Image1: ",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/webp;base64,%s" % (img1),
                    },
                },
                {
                    "type": "text",
                    "text": ", Image2:",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/webp;base64,%s" % (img2),
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
                    "text": "Give you two images, find there shared attributes: Image1:",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/webp;base64,%s" % (img1),
                    },
                },
                {
                    "type": "text",
                    "text": ", Image2:",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/webp;base64,%s" % (img2),
                    },
                },
                {
                    "type": "text",
                    "text": ".\n Your response should only include shared attributes in the following options.\n Options: ['wooden', 'metal', 'flying', 'ripe', 'fresh', 'natural', 'cooked', 'painted', 'rusty', 'furry'].",
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
    



memory_base = "Before this question, you have learnt that related pictures may have the following attributes:\n"
instruct_question = "Based on these knowledge, answer the following quesion:\n"
memory = {}
attr_name = ['wooden', 'metal', 'flying', 'ripe', 'fresh', 'natural', 'cooked', 'painted', 'rusty', 'furry']

given_weight = 1.0
forget_weight = 0.2
def update_memory(answer, obj):
    for attr in attr_name:
        if attr in answer:
            if attr not in memory.keys():
                memory[attr] = {}
            for object in obj:
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
    
def oragnize_memory(mode):
    if mode == "nlmem":
        res = ""
        for attr, object in memory.items():
            objects = ""
            for obj in object:
                objects = objects + obj + ","
            if len(objects) > 0:
                objects = objects[:-1]
            res = res + objects + " has " + attr + " attribute" + ".\n"
            
        return memory_base + res + instruct_question
    
    elif mode == "strmem":
        structured_memory = {}
        for attr, object in memory.items():
            structured_memory[attr] = list(object.keys())

        return f"Given the memory{str(structured_memory)}, please answer the following question.\n"
    
    return ""
paired_attrs = [[9, 2], [4, 6], [3, 5], [7, 8]]

if __name__ == "__main__":
    args = parse_arguments()
    logging.basicConfig(filename=f"OCL_paired_attr_{args.mode}_{args.number}.log", level=logging.INFO)
    for paired_attr in paired_attrs:
        logging.info(f"Paired_Attr: {attr_name[paired_attr[0]]} & {attr_name[paired_attr[1]]} BEGIN")
        step = []
        for epoch in range(100):  
            data = data_loader("./../../data/OCL_test_pics", paired_attr)
            memory = {}
            attr_bank = [attr_name[paired_attr[0]], attr_name[paired_attr[1]]]
            usage = 0
            for attri in attr_bank:
                query_image, correct_image, false_image= data.get_data(attri)
                query_image_encoded = encode_image(data.process_img(query_image))
                correct_image_encoded = encode_image(data.process_img(correct_image))
                if args.mode != "nomem":
                    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=get_attr_prompt(query_image_encoded, correct_image_encoded))
                    response = response.json()
                    if "usage" in response.keys():
                        prompt = response['choices'][0]['message']['content']
                        update_memory(prompt, [query_image['object']['obj'], correct_image['object']['obj']])
                        usage += response['usage']['total_tokens']
                
                if args.number == 3:
                    for tt in range(2):
                        for attri in attr_bank:
                            query_image, correct_image, false_image= data.get_data(attri)
                            query_image_encoded = encode_image(data.process_img(query_image))
                            correct_image_encoded = encode_image(data.process_img(correct_image))
                            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=get_attr_prompt(query_image_encoded, correct_image_encoded))
                            response = response.json()
                            if "usage" in response.keys():
                                prompt = response['choices'][0]['message']['content']
                                update_memory(prompt, [query_image['object']['obj'], correct_image['object']['obj']])
                                usage += response['usage']['total_tokens']
                
            attr = random.choice(attr_bank)
            
            for i in range(100):
                
                another_attr = attr
                attr_new = attr
                if i and i % 5 == 0: 
                    if len(data.query_attr) > 0:
                        while another_attr == attr:
                            another_attr = random.choice(attr_bank)
                            if another_attr in data.query_attr:
                                attr_new = another_attr
                attr = attr_new
                query_image = correct_image
                correct_image, false_image = data.get_next(attr)
                query_image_encoded = encode_image(data.process_img(query_image))
                correct_image_encoded = encode_image(data.process_img(correct_image))
                false_image_encoded = encode_image(data.process_img(false_image))
                dir = random.randint(0, 1)
                correct_ans = 0
                logging.info(f"Iteration{i}, memory:{oragnize_memory(args.mode)}")
                if dir:
                    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=get_payload(oragnize_memory(args.mode), query_image_encoded, correct_image_encoded, false_image_encoded))
                    correct_ans = 1
                else:
                    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=get_payload(oragnize_memory(args.mode), query_image_encoded, false_image_encoded, correct_image_encoded))
                    correct_ans = 2
                response = response.json()
                if "usage" in response.keys():
                    usage += response['usage']['total_tokens']
                else:
                    i -= 1
                    pass
                answer = find_number_after_image(response['choices'][0]['message']['content'])
                logging.info(f"Iteration{i}, gpt4v:{answer}, expected:{correct_ans}, attr:{attr}")
                if answer != correct_ans:
                    print(f"epoch{epoch+1}: answer: {i}")
                    step.append(i)
                    break
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=get_attr_prompt(query_image_encoded, correct_image_encoded))
                response = response.json()
                if "usage" in response.keys():
                        prompt = response['choices'][0]['message']['content']
                        update_memory(prompt, [query_image['object']['obj'], correct_image['object']['obj']])
                        usage += response['usage']['total_tokens']
                logging.info(f"Iteration{i}, Overall usage{usage}")
            

        logging.info(f"mean: {np.mean(step)}, max: {np.max(step)}, {step}")
        logging.info(f"Paired_Attr: {attr_name[paired_attr[0]]} & {attr_name[paired_attr[1]]} END")
