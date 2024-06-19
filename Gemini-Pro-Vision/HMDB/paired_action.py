import pathlib
import textwrap
import re
import logging
import numpy as np
import google.generativeai as genai
from google.generativeai.types.generation_types import BlockedPromptException
from IPython.display import display
from IPython.display import Markdown
from data_loader import *
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--mode', type=str, required=True, help='nomem, strmem, nlmem')
    args = parser.parse_args()

    return args

def result(text):
  text = text.replace('?', '  *')
  return text

GOOGLE_API_KEY="your api key here" # replace this with your own api key

genai.configure(api_key= f"{GOOGLE_API_KEY}", transport="rest")

model = genai.GenerativeModel('gemini-pro-vision')


data = data_loader("./../data/refined_HMDB")

def get_payload(memory, query_img, img1, img2):
    return [f"Instruction: {memory} Determine the relationship between the original image and the candidate images, and select the images with the same action as the original image.\n", query_img, "Candidate images: Image1:",img1, ", Image2:",img2, "Your response should be direct and exclusively only include one of the following items.\n Options: [Image1, Image2]."]
def get_attr(img1, img2):
    return ["Give you two images, find there shared action: Image1:", img1, ", Image2:", img2, ".\n Your response should only include one shared action in the following options.\n Options: ['brush_hair','clap', 'dive', 'shake_hands','hug' ,'sit','smoke','eat']."]

def find_number_after_image(s):
    match = re.search(r'Image(\d+)', s)
    if match:
        return int(match.group(1))
    else:
        return None


memory_base = "Before this question, you have learnt that related pictures may have the following actions:\n"
instruct_question = "Based on these knowledge, answer the following question:\n"
memory = {}
attr_name = ['brush_hair','clap', 'dive', 'shake_hands','hug' ,'sit','smoke','eat']
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
    
    
def organize_memory(mode):
    if mode == "strmem":
        structured_memory = {}
        for attr, object in memory.items():
            structured_memory[attr] = list(object.keys())
        return f"Given the memory{str(structured_memory)}, please answer the following question.\n"
    elif mode == "nlmem":
        res = ""
        for attr, object in memory.items():
            objects = ""
            for obj in object:
                objects = objects + obj + ","
            if len(objects) > 0:
                objects = objects[:-1]
            res = res + objects + " has " + attr + " action" + ".\n"
        return memory_base + res + instruct_question
    return ""
        

# all the four test paired_attributes in HMDB dataset
paired_attrs = [["brush_hair", "dive"], ["smoke", "eat"], ["clap", "hug"], ["shake_hands", "sit"]]

if __name__ == "__main__":    
    args = parse_arguments()
    logging.basicConfig(filename=f"HMDB_{args.mode}.log", level=logging.INFO)
    for paired_attr in paired_attrs:
        logging.info(f"paired_attr : {paired_attr} BEGIN")
        steps = []
        for epoch in range(0,100):
            print(f"epoch: {epoch+1}")
            attrs = paired_attr
            
            logging.info(f"epoch: {epoch+1}")
            data = data_loader("./../../data/HMDB")
            memory = {}
            now = random.randint(0, 1)
            if args.mode != "nomem":
                attr = attrs[now]
                query_image, correct_image, false_image= data.get_data(attr)
                query_image_encoded = (data.process_img(query_image))
                correct_image_encoded = (data.process_img(correct_image))
                try:
                    response = model.generate_content(get_attr(query_image_encoded, correct_image_encoded), stream=False)
                    response.resolve()
                    prompt = result(response.text)
                except ValueError:
                    pass
                else:
                    update_memory(prompt, data.data[query_image]['objects'] + data.data[correct_image]['objects'])

            for i in range(100):
                if i % 5 == 0:
                    now = now ^ 1
                    attr = attrs[now]
                    query_image , correct_image, false_image = data.get_data(attr)
                else:
                    query_image = correct_image
                    correct_image, false_image = data.get_next(attr)
                query_image_encoded = (data.process_img(query_image))
                correct_image_encoded = (data.process_img(correct_image))
                false_image_encoded = (data.process_img(false_image))
                dir = random.randint(0, 1)
                correct_ans = 0
                try:
                    if dir: 
                        response = model.generate_content(get_payload(organize_memory(args.mode), query_image_encoded, correct_image_encoded, false_image_encoded), stream=False)
                        correct_ans = 1
                    else:
                        response = model.generate_content(get_payload(organize_memory(args.mode), query_image_encoded, false_image_encoded, correct_image_encoded), stream=False)
                        correct_ans = 2
                    response.resolve()
                    answer = find_number_after_image(result(response.text))   
                except ValueError:
                    i -= 1
                    continue
                logging.info(f"Iteration{i}, gemini:{answer}, expected:{correct_ans}, attr:{attr}")
                if answer != correct_ans:
                    steps.append(i)
                    print(i)
                    break
                if args.mode != "nomem":
                    try:
                        response = model.generate_content(get_attr(query_image_encoded, correct_image_encoded), stream=False)
                        response.resolve()
                        prompt = result(response.text)
                    except ValueError:
                        pass
                    else:
                        update_memory(prompt, data.data[query_image]['objects'] + data.data[correct_image]['objects'])
            logging.info(f"final steps:{steps}")
            
        
        logging.info(f"mean: {np.mean(steps)}, max: {np.max(steps)}")
        
        logging.info(f"paired_attr : {paired_attr} END")