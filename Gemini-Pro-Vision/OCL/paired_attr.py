import random
import re
from data_loader_two import *
import logging
import textwrap
import re
import logging
import numpy as np
import google.generativeai as genai
from google.generativeai.types.generation_types import BlockedPromptException
from IPython.display import Markdown
from data_loader_two import *
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--mode', type=str, required=True, help='nomem, strmem, nlmem')
    parser.add_argument('--number', type=int, required=False, default=1, help='number of shot: 1 or 3')
    args = parser.parse_args()

    return args

def get_payload(memory, query_img, img1, img2):
    return [f"Instruction: {memory} Determine the relationship between the original image and the candidate images, and select the images with the same action as the original image.\n", query_img, "Candidate images: Image1:",img1, ", Image2:",img2, "Your response should be direct and exclusively only include one of the following items.\n Options: [Image1, Image2]."]
   
def get_attr(img1, img2):
    return ["Give you two images, find there shared attributes: Image1:", img1, ", Image2:", img2, ".\n Your response should only include one shared action in the following options.\n Options: [ 'metal', 'ripe', 'fresh', 'natural', 'cooked', 'painted', 'rusty', 'furry']."]

def result(text):
  text = text.replace('?', '  *')
  return text

GOOGLE_API_KEY="AIzaSyBxFBvM-2fuK0-UnMIG9MCj_PyIGPibaJ8"

genai.configure(api_key= f"{GOOGLE_API_KEY}", transport="rest")

model = genai.GenerativeModel('gemini-pro-vision')


    
def find_number_after_image(s):
    match = re.search(r'Image(\d+)', s)
    if match:
        return int(match.group(1))
    else:
        return None
    


memory_base = "Before this question, you have learnt that related pictures may have the following attributes:\n"
instruct_question = ".Based on these knowledge, answer the following question:\n"
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
            res = res + objects + " has " + attr + " attribute" + ".\n"
        return memory_base + res + instruct_question    
    
    return ""    

paired_attrs = [[9, 2], [4, 6], [3, 5], [7, 8]]



if __name__ == "__main__":

    args = parse_arguments()
    logging.basicConfig(filename=f"OCL_paired_attr_{args.mode}_{args.number}.log", level=logging.INFO)
    for paired_attr in paired_attrs:
        logging.info(f"Paired_Attr: {attr_name[paired_attr[0]]} & {attr_name[paired_attr[1]]} BEGIN")
        step = []
        for epoch in range(5):  
            data = data_loader("./../../data/OCL_test_pics", paired_attr)
            #attr = random.choice(attr_name)
            memory = {}
            attr_bank = [attr_name[paired_attr[0]], attr_name[paired_attr[1]]]
            attr = random.choice(attr_bank)
            query_image, correct_image, false_image= data.get_data(attr)
            query_image_encoded = (data.process_img(query_image))
            correct_image_encoded = (data.process_img(correct_image))
            if args.mode != "nomem":
                try:
                    response = model.generate_content(get_attr(query_image_encoded, correct_image_encoded), stream=False)
                    response.resolve()
                except BlockedPromptException:
                    pass
                else:
                    prompt = result(response.text)
                    update_memory(prompt, [query_image['object']['obj'], correct_image['object']['obj']])
                if args.number == 3:
                    attr = random.choice(attr_bank)
                    query_image, correct_image, false_image= data.get_data(attr)
                    query_image_encoded = (data.process_img(query_image))
                    correct_image_encoded = (data.process_img(correct_image))
                    try:
                        response = model.generate_content(get_attr(query_image_encoded, correct_image_encoded), stream=False)
                        response.resolve()
                    except BlockedPromptException:
                        pass
                    else:
                        prompt = result(response.text)
                        update_memory(prompt, [query_image['object']['obj'], correct_image['object']['obj']])
                                    
            for i in range(100):    
                another_attr = attr
                attr_new = attr
                if i % 5 == 0: 
                    while another_attr == attr:
                        another_attr = random.choice(attr_bank)
                        if another_attr in data.query_attr:
                            attr_new = another_attr
                    attr = attr_new    
                    query_image = correct_image
                    correct_image, false_image = data.get_next(attr)  
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
                        response = model.generate_content(get_payload(oragnize_memory(args.mode), query_image_encoded, correct_image_encoded, false_image_encoded), stream=False)
                        correct_ans = 1
                    else:
                        response = model.generate_content(get_payload(oragnize_memory(args.mode), query_image_encoded, false_image_encoded, correct_image_encoded), stream=False)
                        correct_ans = 2
                    response.resolve()
                except BlockedPromptException:
                    i -= 1
                    continue
                answer = find_number_after_image(result(response.text))   
                logging.info(f"Iteration{i}, gemini:{answer}, expected:{correct_ans}, attr:{attr}")
                if answer != correct_ans:
                    print(f"epoch{epoch+1}: answer: {i}")
                    step.append(i)
                    break
                
                if args.mode != "nomem":
                    try:
                        response = model.generate_content(get_attr(query_image_encoded, correct_image_encoded), stream=False)
                        response.resolve()
                    except BlockedPromptException:
                        pass
                    else:
                        prompt = result(response.text)
                        update_memory(prompt, [query_image['object']['obj'], correct_image['object']['obj']])
                        
        logging.info(f"mean: {np.mean(step)}, max: {np.max(step)}, {step}")
        logging.info(f"Paired_Attr: {attr_name[paired_attr[0]]} & {attr_name[paired_attr[1]]} BEGIN")
