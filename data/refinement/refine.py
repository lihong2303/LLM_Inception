import re
import logging
import re
import logging
import google.generativeai as genai
from google.generativeai.types.generation_types import BlockedPromptException
import argparse
import os
from PIL import Image
import pickle
import base64
import requests
from io import BytesIO

def parse_arguments():
    parser = argparse.ArgumentParser(description='Data refinement by Gemini-1.5-Flash and GPT-4o.')
    parser.add_argument('--type', type=str, required=True, help='attribute, affordance, action')
    argss = parser.parse_args()

    return argss

def get_payload_Gemini(attr, query_img):
    return [f"Analyze the provided image and determine if it contains the {args.type}: {attr}.\n Image:", query_img, "Provide only 'Yes' or 'No' as the answer, without any additional explanation."]

def result(text):
  text = text.replace('?', '  *')
  return text

GOOGLE_API_KEY="your api key here"

genai.configure(api_key= f"{GOOGLE_API_KEY}", transport="rest")
    
model = genai.GenerativeModel('gemini-1.5-flash')

    
def find_yes_or_no(s):
    match = re.search(r'\b(yes|no)\b', s, re.IGNORECASE)
    if match:
        return match.group(1).lower() == 'yes'
    else:
        return None
    
def process_img(img, bbox):
    # convert to RGB, resize to 336x336
    image = Image.open(img)
    if image.mode == 'RGBA' or image.mode == 'P':
        # Convert the image to RGB
        image = image.convert('RGB')
    
    if bbox:
        image = image.crop(bbox)
    image = image.resize((336, 336))
    return image


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


def get_payload_GPT(query_img, attr):
    payload = {
    "model": "gpt-4o",
    "messages": [
        {
        "role": "user",
        "content": [
                {
                    "type": "text",
                    "text": f"Analyze the provided image and determine if it contains the {args.type}: {attr}.\n Image:",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/webp;base64,%s" % (encode_image(query_img)),
                    },
                },
                {
                    "type": "text",
                    "text": "Provide only 'Yes' or 'No' as the answer, without any additional explanation.",
                },
        ]
        }
    ],
    "max_tokens": 50
    }
    return payload

def judge(img_path, attr, bbox):
    
    attr = ''.join([c for c in attr if c.islower()])
    try:
        response = model.generate_content(get_payload_Gemini(attr, process_img(img_path, bbox)), stream=False)
        response.resolve()
        res = result(response.text)
        
    except BlockedPromptException:
        return -1
    except ValueError:
        return -1

    if find_yes_or_no(res) == True:
        return 1
    if find_yes_or_no(res) == False:
        return 0
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=get_payload_GPT(process_img(img_path, bbox), attr))
    response = response.json()
    if "usage" in response.keys():
        answer = find_yes_or_no(response['choices'][0]['message']['content'])
        
    if answer == True:
        return 1
    
    return 0
    
if __name__ == "__main__":

    global args
    args = parse_arguments()
    
    logging.basicConfig(filename=f"{args.type}_refine.log",level=logging.INFO)
        
    if args.type == "attribute":
        paired_options = ['fresh-cooked', 'natural-ripe', 'furry-metal', 'painted-rusty']
    elif args.type == "affordance":
        paired_options = ['sit-write', 'push-carry', 'cut-clean', 'open-break', 'cut-close']
    elif args.type == "action":
        paired_options = ['drive-11.5_dress-41.1.1-1-1', 'shake-22.3-2_cut-21.1-1', 'cooking-45.3_build-26.1', 'run-51.3.2_hit-18.1']
    
    if args.type != "action":
        pic_folder = "./../data/OCL"
    else:
        pic_folder = "./../data/Pangea"

    
        
    if args.type != "action":
        with open ("./../data/OCL_annot_test.pkl", "rb") as f:
            data = pickle.load(f)
        with open(f"./../data/OCL_selected_test_{args.type}.pkl", "rb") as f:
            cross = pickle.load(f)
    else:
        with open(f"./../data/pangea_test_new.pkl", "rb") as f:
            cross = pickle.load(f)
        with open ("./../data/B123_test_KIN-FULL_with_node.pkl", "rb") as f:
            data = pickle.load(f)
            
    correct_result = {}
    not_sure = {}
    
    for paired_option in paired_options:    
        logging.info(f"Gemini-1.5-Flash refine: {paired_option} BEGIN")
        if args.type == "action":
            attr_bank = paired_option.split('_')
        else:
            attr_bank = paired_option.split('-')
            
        correct_result[paired_option] = {}
        not_sure[paired_option] = {}
        for attr in attr_bank:
            correct_data = []
            not_sure_data = []
            idx = 0
            for item in cross['selected_paired_pkl'][paired_option][attr]:
                if args.type == "action":
                    img_path = os.path.join(pic_folder, data[item][1])
                else:
                    img_path = os.path.join(pic_folder, data[item[0]]['name'])
                bbox = None
                if args.type != "action":
                    bbox = data[item[0]]['objects'][item[1]]['box']                    
                res = judge(img_path, attr, bbox)
                if res == 1:
                    correct_data.append(item)
                elif res == -1:
                    not_sure_data.append(item)
                logging.info(f"{attr}, {idx}, {res}")
                idx += 1
                
            correct_result[paired_option][attr] = correct_data
            not_sure[paired_option][attr] = not_sure_data
        
        correct_data = []
        not_sure_data = []
        idx = 0
        for item in cross['selected_paired_pkl'][paired_option][paired_option]:
            if idx <= 1070:
                idx += 1
                continue
            if args.type == "action":
                img_path = os.path.join(pic_folder, data[item][1])
            else:
                img_path = os.path.join(pic_folder, data[item[0]]['name'])
            bbox = None
            if args.type != "action":
                bbox = data[item[0]]['objects'][item[1]]['box']    
                            
            res1 = judge(img_path, attr_bank[0], bbox)
            res2 = judge(img_path, attr_bank[1], bbox)
            
            if res1 == 1 and res2 == 1:
                correct_data.append(item)
            elif res1 == -1 or res2 == -1:
                not_sure_data.append(item)
            logging.info(f"{paired_option}, {idx}, {res1}, {res2}")
            idx += 1
            
        correct_result[paired_option][paired_option] = correct_data
        not_sure[paired_option][paired_option] = not_sure_data
        
        logging.info(f"Gemini-1.5-Flash refine: {paired_option} END")
        
    if args.type != "action":
        with open(f"./../data/OCL_selected_test_{args.type}_refined.pkl", "wb") as f:
            pickle.dump(correct_result, f)
        with open(f"./../data/OCL_selected_test_{args.type}_not_sure.pkl", "wb") as f:
            pickle.dump(not_sure, f)
    else:
        with open(f"./../data/pangea_test_new_refined.pkl", "wb") as f:
            pickle.dump(correct_result, f)
        with open(f"./../data/pangea_test_new_not_sure.pkl", "wb") as f:
            pickle.dump(not_sure, f)
    