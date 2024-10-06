import random
import re
from data_loader import *
import logging
import re
import logging
import numpy as np
import google.generativeai as genai
from google.generativeai.types.generation_types import BlockedPromptException
import argparse

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
            
attr_name_mapping = {'run':'run-51.3.2', 'hit':'hit-18.1', 'drive':'drive-11.5', 'dress':'dress-41.1.1-1-1', 'shake':'shake-22.3-2', 'cut':'cut-21.1-1', 'cooking':'cooking-45.3', 'build':'build-26.1'}

def get_payload(memory, query_img, img1, img2):
    # print([f"Instruction:{memory} Determine the relationship between the original image and the candidate images, and select the images with the same {args.type} as the original image.\n", query_img, "Candidate images: Image1:",img1, ", Image2:",img2, "Your response should be direct and exclusively only include one of the following items.\n Options: [Image1, Image2]."])
    return [f"Instruction:{memory} Determine the relationship between the original image and the candidate images, and select the images with the same {args.type} as the original image.\n", query_img, "Candidate images: Image1:",img1, ", Image2:",img2, "Your response should be direct and exclusively only include one of the following items.\n Options: [Image1, Image2]."]
   
def get_attr(img1, img2): 
    return [f"Generate the shared {args.type} between the original image and selected images.\n: Question: Original image:", img1, ", Selected image:", img2, f".\n Your response should include all shared {args.type} in the following options.\n Options: {attr_name[args.type]}."]

def result(text):
  text = text.replace('?', '  *')
  return text

GOOGLE_API_KEY="your api key here"

genai.configure(api_key= f"{GOOGLE_API_KEY}", transport="rest")
    
model = genai.GenerativeModel('gemini-1.5-flash')
    
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
forget_weight = 0
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

def process_img(data, img):
    
    if args.type == 'action':
        name = data[img][1]
        folder_path = "./../data/Pangea"
    else:
        name = data[img[0]]['name']
        folder_path = "./../data/OCL"

    image = Image.open(os.path.join(folder_path, name))
    
    if image.mode == 'RGBA' or image.mode == 'P':
        image = image.convert('RGB')
    if args.type == 'action':
        image = image.resize((336, 336))
    else:
        box = data[img[0]]['objects'][img[1]]['box']
        image = image.crop(box).resize((336, 336))
    return image
       
if __name__ == "__main__":

    global args
    global memory_base
    args = parse_arguments()
    
    memory_base = f"Before this question, you have learnt that related pictures may have the following {args.type}:\n"
    
    if args.type != 'action':
        logging.basicConfig(filename=f"OCL_single_{args.type}_{args.mode}_{args.number}.log", level=logging.INFO)
    else:
        logging.basicConfig(filename=f"Pangea_single_{args.type}_{args.mode}_{args.number}.log", level=logging.INFO)

    for attr in attr_name[args.type]:
        if attr == "imprint":
            attr = "write"
        if args.type == 'action':
            original_attr = attr
            attr = attr_name_mapping[attr]
            
        logging.info(f"Attr: {attr} BEGIN")
        memory = {}
        step = []
        if args.type != 'action':
            data = read_pkl_file("./../data/OCL_annot_test.pkl")
            cross = read_pkl_file(f"./../data/OCL_selected_test_{args.type}_refined_new_1.pkl")
        else:
            cross = read_pkl_file(f"./../data/pangea_test_refined_new.pkl")
            data = read_pkl_file(f"./../data/B123_test_KIN-FULL_with_node.pkl")
        negative_cross = cross['negative_pkl']
        cross = cross['selected_individual_pkl'][attr]
        for i in range(args.number):
            if args.type != 'action':
                update_memory(attr,[data[cross[i][0]]['objects'][cross[i][1]]['obj'], data[cross[i + 1][0]]['objects'][cross[i + 1][1]]['obj']])
            else:
                update_memory(original_attr, [data[cross[i]][2], data[cross[i+1]][2]])
                    
        query_image = cross[1]
        tot_test = 0
        correct_test = 0
        
        # tot_evidence = 0
        # correct_evidence = 0
        
        for i in range(args.number + 1, min(502, len(cross))):
            query_image = cross[i]
            correct_image = random.choice(cross)
            false_image = random.choice(negative_cross)
            query_image_encoded = (process_img(data, query_image))
            correct_image_encoded = (process_img(data, correct_image))
            false_image_encoded = (process_img(data, false_image))
            dir = random.randint(0, 1)
            correct_ans = 0
            logging.info(f"Iteration{i}, memory: {organize_memory(args.mode)}")
            try:
                if dir: 
                    response = model.generate_content(get_payload(organize_memory(args.mode), query_image_encoded, correct_image_encoded, false_image_encoded), stream=False)
                    correct_ans = 1
                else:
                    response = model.generate_content(get_payload(organize_memory(args.mode), query_image_encoded, false_image_encoded, correct_image_encoded), stream=False)
                    correct_ans = 2
                response.resolve()
                answer = find_number_after_image(result(response.text))
            except BlockedPromptException:
                continue
            except ValueError:
                continue
            if answer == None:
                continue
            logging.info(f"Iteration{i}, gemini:{answer}, expected:{correct_ans}, attr:{attr}, Query:{query_image}, Correct:{correct_image}, False:{false_image}")
            if answer == correct_ans:
                correct_test += 1
            tot_test += 1
            # if original_attr in attr_name[args.type][5:]:
            #     try:
            #         response = model.generate_content(get_attr(query_image_encoded, correct_image_encoded), stream=False)
            #         response.resolve()
            #         prompt = result(response.text)
            #     except BlockedPromptException:
            #         pass
            #     except ValueError:
            #         pass
            #     logging.info(f"Prompt: {prompt}")
            #     if original_attr in prompt:
            #         correct_evidence += 1
            #     tot_evidence += 1
            
            if i % 50 == 0 and i > 0:
                # if original_attr in attr_name[args.type][5:]:
                #     logging.info(f"Iter:{i}, Test Accuracy: {correct_test/tot_test}, Evidence Accuracy: {correct_evidence/tot_evidence}")
                # else:
                logging.info(f"Iter:{i}, Test Accuracy: {correct_test/tot_test}")
        # if original_attr in attr_name[args.type][5:]:
        #     logging.info(f"Attr:{attr}, Test Accuracy: {correct_test/tot_test}, Evidence Accuracy: {correct_evidence/tot_evidence}")
        # else:
        logging.info(f"Attr:{attr}, Test Accuracy: {correct_test/tot_test}")
        logging.info(f"Attr: {attr} END")
