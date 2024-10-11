import os
import re
import torch
import random
import warnings
import argparse
import logging
import time
from PIL import Image
from collections import Counter
import numpy as np
from torch.utils.data import DataLoader
from Model.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from Model.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor

from qwen_vl_utils import process_vision_info

# LLaVa-Onevision
from Model.llava_OneVision.model.builder import load_pretrained_model_OneVision
from Model.llava_OneVision.mm_utils import get_model_name_from_path_OneVision

from Model.llava_OneVision.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IGNORE_INDEX
)
from Utils.preliminary_llava_onevision import preprocess_qwen

from Model.llava_OneVision.conversation import conv_templates,conv_qwen,SeparatorStyle

from Model.llava_OneVision.mm_utils import (process_images,
                         tokenizer_image_token)

from transformers import AutoTokenizer, AutoProcessor,AutoModel

from Model.mPLUG_Owl3_7B_240728.modeling_mplugowl3 import mPLUGOwl3Model

from Data import (OCLDataset,
                  HMDB51Dataset,
                  PangeaDataset,
                  OCL_data_collate,
                  HMDB_data_collate,
                  Pangea_data_collate,
                  )


warnings.filterwarnings("ignore", category=UserWarning)

llava_onevision_checkpoint_path = "lmms-lab/llava-onevision-qwen2-7b-ov"
qwen2_vl_model_path = "Qwen/Qwen2-VL-7B-Instruct"
mplug3_model_path = "mPLUG/mPLUG-Owl3-7B-240728"

mapping_dataset_directory = {'ActvityNet_hico_style_batch1':'ActivityNet_hico_batch1','charadesEgo_hico_style':'charadesego_frame', 'HAG_hico_style_new':'hag_frame','HACS_hico_style':'hacs_frame','kinetics_hico_style':'kinetics_dataset/k700-2020/train'}
map_dict = {"hit":"hit-18.1",
            "push":"push-12",
            "run":"run-51.3.2",
            "dress":"dress-41.1.1-1-1",
            "drive":"drive-11.5",
            "cooking":"cooking-45.3",
            "throw":"throw-17.1-1",
            "build":"build-26.1",
            "shake":"shake-22.3-2",
            "cut":"cut-21.1-1"}

def get_logger(logger_name,logger_dir=None,log_name=None,is_mute_logger=False):
    logger = logging.getLogger(logger_name)
    logger.handlers.clear() 

    if is_mute_logger:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    hterm = logging.StreamHandler()
    hterm.setFormatter(formatter)
    hterm.setLevel(logging.INFO)
    logger.addHandler(hterm)

    if logger_dir is not None:
        if log_name is None:
            logger_path = os.path.join(logger_dir,f"{logger_name}.log")
        else:
            logger_path = os.path.join(logger_dir,log_name)
        hfile = logging.FileHandler(logger_path) 
        hfile.setFormatter(formatter)
        hfile.setLevel(logging.INFO)
        logger.addHandler(hfile)
    return logger


def compute_metrics(preds):
        
    accu_list = preds
    
    max_count = np.max(accu_list)
    min_count = np.min(accu_list)
    
    mean_count = np.mean(accu_list)
    
    result = {"Max":max_count,
                "Min":min_count,
                "mean_count":mean_count}
    
    return result

def get_pangea_img(file_dir, data_item,object_to_id):
    
    if data_item[0] in mapping_dataset_directory.keys():
        dataset = mapping_dataset_directory[data_item[0]]
    else:
        dataset = data_item[0]
    
    image_path = file_dir + '/' + dataset + '/' + data_item[1]
    
    cur_objects = ' '.join(data_item[2])
    node_labels = data_item[3]
    node_labels_id = [object_to_id[nod_lab] for nod_lab in node_labels]
    
    cur_image = Image.open(image_path)
    cur_image = cur_image.resize((336,336)).convert('RGB')
    
    return cur_objects, node_labels, node_labels_id, cur_image 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',type=str,default=None)
    parser.add_argument('--data_type',type=str,default="ocl_attr_data")
    parser.add_argument('--model_type',type=str,default="llava-onevision")
    parser.add_argument('--attr_constraint',type=str,default=None)
    parser.add_argument('--prompt_type',type=str,default=None)
    parser.add_argument('--expt_dir',type=str,default="logs/test_experiment")
    parser.add_argument('--few_shot_num',type=int,default=None)
    args= parser.parse_args()
    return args


def QA_with_Qwen2vl(model,processor,prompt,input_images):
    """
    """
    conversation = []
    split_prompt = prompt.split(DEFAULT_IMAGE_TOKEN)
    
    for sp_idx,sp_pro in enumerate(split_prompt):
        
        conversation.append({"type": "text", "text":split_prompt[sp_idx]})
        if sp_idx < len(input_images):
            conversation.append({"type": "image", "image":input_images[sp_idx]})
    messages = messages = [{"role": "user", "content": conversation}]
    
    text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt")
    inputs = inputs.to("cuda")
    
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    
    generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    return output_text

def QA_with_LLavaOneVision(model,tokenizer,image_preprocess,prompt,input_images):
    """
    """ 
    with torch.no_grad():
        image_sizes = [x.size for x in input_images]
        preprocessed_image = image_preprocess.preprocess(input_images, return_tensors="pt")["pixel_values"].to("cuda",torch.float16)
        input_ids = preprocess_qwen([{'from': 'human','value': prompt},{'from': 'gpt','value': None}], tokenizer, has_image=True).to("cuda")
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=preprocessed_image,
                image_sizes=image_sizes,
                do_sample=True,
                max_new_tokens=1024,
                use_cache=True,
                pad_token_id = model.config.eos_token_id,
                return_dict_in_generate=True, 
                output_attentions=True)
                
        response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
    return response

def QA_with_mplug3(model,tokenizer,processor,prompt,input_images):
    """
    """
    prompt = prompt.replace(DEFAULT_IMAGE_TOKEN,'<|image|>')
    messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": ""}]
    
    inputs = processor(messages, images=input_images)
    inputs.to('cuda')

    inputs.update({
        'tokenizer': tokenizer,
        'max_new_tokens':512,
        'decode_text':True,
    })

    response = model.generate(**inputs)[0].strip()
    return response

def QA_with_MOE(qwen2vl_model,qwen2vl_processor,llava_onevision_model,llava_onevision_tokenizer,llava_onevision_image_preprocess,mplug3_model,mplug3_tokenizer,mplug3_processor,prompt,input_images):
    """
    """
    mplu3_response = QA_with_mplug3(mplug3_model,mplug3_tokenizer,mplug3_processor,prompt,input_images)
    
    llava_onevision_response = QA_with_LLavaOneVision(llava_onevision_model,llava_onevision_tokenizer,llava_onevision_image_preprocess,prompt,input_images)
    
    qwen2vl_response = QA_with_Qwen2vl(qwen2vl_model,qwen2vl_processor,prompt,input_images)
    
    mixed_response = llava_onevision_response + " " + qwen2vl_response + " " + mplu3_response
    mixed_response = re.sub(r'[^\w\s]', '', mixed_response)
    mixed_response = mixed_response.strip().split()
    
    mixed_response = Counter(mixed_response)
    
    max_count = mixed_response.most_common(1)[0][1]
    
    most_common_words = [word for word, count in mixed_response.items() if count == max_count]
    
    
    selected_word = random.choice(most_common_words)
    
    return selected_word


def main():
    
    args = parse_args()
    expt_name = args.attr_constraint + "_" + args.prompt_type + "_" + str(args.few_shot_num)
    expt_dir = os.path.join(args.expt_dir,args.model_type,args.data_type)
    save_dir = os.path.join(expt_dir,f"{expt_name}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger = get_logger("train_logger",logger_dir=save_dir)

    logger.info(vars(args))
    
    attr_constraint = args.attr_constraint
    few_shot_num = args.few_shot_num
    prompt_type = args.prompt_type

    if args.model_type == "llava-onevision":
        llava_onevision_tokenizer, llava_onevision_model, llava_onevision_image_processor, llava_onevision_context_len = load_pretrained_model_OneVision(llava_onevision_checkpoint_path,
                                                                                                                    model_base=None,
                                                                                                                    model_name=get_model_name_from_path_OneVision(llava_onevision_checkpoint_path),
                                                                                                                    device_map="cuda",
                                                                                                                    attn_implementation="eager")
    elif args.model_type == "qwen2vl":
        qwen_processor = Qwen2VLProcessor.from_pretrained(qwen2_vl_model_path)
        qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(qwen2_vl_model_path, torch_dtype=torch.float16, device_map="cuda",attn_implementation="eager")
        qwen_model = qwen_model.half()
    elif args.model_type == "mplug3":
        mplug3_model = mPLUGOwl3Model.from_pretrained(mplug3_model_path, attn_implementation='sdpa', trust_remote_code=True, torch_dtype=torch.bfloat16)
        mplug3_tokenizer = AutoTokenizer.from_pretrained(mplug3_model_path)
        mplug3_model.eval().cuda()
        mplug3_processor = mplug3_model.init_processor(mplug3_tokenizer)
    elif args.model_type == "moe":
        mplug3_model = mPLUGOwl3Model.from_pretrained(mplug3_model_path, attn_implementation='sdpa', trust_remote_code=True, torch_dtype=torch.bfloat16)
        mplug3_tokenizer = AutoTokenizer.from_pretrained(mplug3_model_path)
        mplug3_model.eval().cuda()
        mplug3_processor = mplug3_model.init_processor(mplug3_tokenizer)
        
        llava_onevision_tokenizer, llava_onevision_model, llava_onevision_image_processor, llava_onevision_context_len = load_pretrained_model_OneVision(llava_onevision_checkpoint_path,
                                                                                                                    model_base=None,
                                                                                                                    model_name=get_model_name_from_path_OneVision(llava_onevision_checkpoint_path),
                                                                                                                    device_map="cuda")
        qwen_processor = Qwen2VLProcessor.from_pretrained(qwen2_vl_model_path)
        qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(qwen2_vl_model_path, torch_dtype=torch.float16, device_map="cuda")
        
        
    if args.data_type in ["ocl_attr_data","ocl_aff_data"]:
    
        eval_dataset = OCLDataset("Data","test",attr_constraint = attr_constraint,train_multistep = False,data_type = args.data_type)
        data_collator = OCL_data_collate
    elif args.data_type == "pangea_data":
        eval_dataset = PangeaDataset("Data","test",chain_constraint = attr_constraint)
        data_collator = Pangea_data_collate

    dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, collate_fn=data_collator)

    # Get the first example
    first_example = next(iter(dataloader))
    idx = first_example['idx']
    file_dir = first_example['file_dir']
    pkl_data= first_example['pkl_data']
    attr = first_example['attr']

    attr_constrain = attr_constraint.split(",")
    if len(attr) == 290:
        attr_constrain = [map_dict[attr_item] for attr_item in attr_constrain]

    attr_constrain_id = [attr.index(attr_item) for attr_item in attr_constrain]

    if len(attr) == 170:
        chain_type = "affordance"
        selected_chain = ['break', 'carry', 'clean','close','cut','eat','open','push','sit','imprint'] # 'write'
    elif len(attr) == 114:
        chain_type = "attribute"
        selected_chain = ['wooden', 'metal', 'flying', 'ripe', 'fresh', 'natural', 'cooked', 'painted', 'rusty', 'furry']
    elif len(attr) == 290:
        chain_type = "action"
        selected_chain = ["hit-18.1","push-12","run-51.3.2","dress-41.1.1-1-1","drive-11.5","cooking-45.3","throw-17.1-1","build-26.1","shake-22.3-2","cut-21.1-1"]
        selected_chain_clean = [sel_cha.split("-")[0] for sel_cha in selected_chain]
            
        selected_chain_id = [attr.index(sel_chain) for sel_chain in selected_chain]
            
    if len(attr_constrain) == 2:
        full_pkl = pkl_data[0]
        pkl_data1 = pkl_data[1]
        pkl_data2 = pkl_data[2]
        pkl_data1_data2 = pkl_data[3]
        negative_data = pkl_data[4]
    else:
        full_pkl = pkl_data[0]
        pkl_data1 = pkl_data[1]
        negative_data = pkl_data[2]

    
    
    if len(attr_constrain) == 2:
        num_iters = max(2000,min(len(pkl_data1),len(pkl_data2)))
    else:
        num_iters = max(2000,len(pkl_data1))
    
    accu_steps_list = []
    
    for each_steps in range(num_iters):
        
        # set initialize index
        if len(attr_constrain) == 2:
            if each_steps > min(len(pkl_data1),len(pkl_data2)) - 1:
                initialize_idx = each_steps % min(len(pkl_data1),len(pkl_data2))
            else:
                initialize_idx = each_steps
            
        else:
            if each_steps > len(pkl_data1) - 1:
                initialize_idx = each_steps % len(pkl_data1)
            else:
                initialize_idx = each_steps
        
        
        if chain_type == "action":
            object_to_id = {attr_item: attr_idx for attr_idx,attr_item in enumerate(attr)}

        memory = {}
        memory_weight = {}

        instruction_token = "Instruction: "
        if prompt_type in ["task_instruction","task_instruction_nomem","task_instruction_chainmem", "task_instruction_nlp", "task_instruction_tuned", "task_instruction_nlp_tuned"]:
            instruction = f"Determine the relationship between the original image and the candidate images, and select the images with the same {chain_type} as the original image.\n"
            question = f"Question: Original image:{DEFAULT_IMAGE_TOKEN}. Candidate images: Image1:{DEFAULT_IMAGE_TOKEN}, Image2:{DEFAULT_IMAGE_TOKEN}. Your response should be direct and exclusively only include one of the following items.\n Options: [Image1, Image2]."

            instruction2 = f"Generate the common {chain_type} between the original image and selected images.\n" 
            question2 = f"Question: Original image:{DEFAULT_IMAGE_TOKEN}. Selected image: {DEFAULT_IMAGE_TOKEN}. Your response should include all shared {chain_type} in the following options.\n Options:"
                        
        accu_step = 0
        examp_step = 0
                
        data_dir_index = file_dir[0]

        # ======= Create Init Image =====
        # first example original image from attribute1
        attrs_2_idx = {attr_item:idx for idx,attr_item in enumerate(attr)}
        true_attr_list =  [attrs_2_idx[attr_idx] for attr_idx in attr_constrain]

        if chain_type == "action":
            cur_img_idx = pkl_data1[initialize_idx]
            
            cur_instance = full_pkl[cur_img_idx]
            
            cur_obj_name, cur_obj_chain, cur_obj_attr, cur_image = get_pangea_img(file_dir[0], cur_instance, object_to_id)
            
        else:
            # select first example
            [cur_img_idx, cur_object_idx] = pkl_data1[initialize_idx]
                    
            cur_obj = full_pkl[cur_img_idx]["objects"][cur_object_idx]
            cur_name_info = full_pkl[cur_img_idx]["name"]
            
            if chain_type == "affordance":
                cur_obj_attr = cur_obj['aff']
            elif chain_type == "attribute":
                cur_obj_attr = cur_obj['attr']
            
            cur_obj_box = cur_obj['box']
            cur_obj_name = cur_obj['obj']
            cur_image_path = os.path.join(file_dir[0],cur_name_info)
            cur_image = Image.open(cur_image_path)
            cur_image = cur_image.crop(cur_obj_box)
            cur_image = cur_image.resize((336,336)).convert('RGB')
            
        label_dict = {0:["Image1", "Image 1"],
                    1:["Image2", "Image 2"]}

        syne_proc = []
        syne_result = True
        syne_log = ""

        accu_attr = 0


        while syne_result and accu_step < 500:
            
            syne_step_log = {}
            syne_step_log['Iter'] = each_steps
            syne_step_log['step'] = accu_step
            if examp_step < len(attr_constrain) * few_shot_num:
                if examp_step == 0:
                    syne_log += 'Example: '
                    
                syne_step_log['orig_img'] = cur_obj_name
                syne_log += cur_obj_name
                syne_log += '->'
            else:
                if accu_step == 0:
                    syne_log += 'Test begining:'
                    
                syne_step_log['orig_img'] = cur_obj_name
                syne_log += cur_obj_name
                syne_log += '->'
            
            # create cand images
            total_num = 2
            true_num = 1
            false_num = 1
            true_count = 0
            false_count = 0
            
            
                
            image_banks = []
            attr_banks = []
            idx_banks = []
            co_attr_label = []
            co_attr_bank = []
            obj_name_banks = []
            while (true_count + false_count) < total_num:
                # select non-repeat indices
                if true_count < true_num:
                    if examp_step < (len(attr_constrain) * few_shot_num):
                        # select example candidate
                        if len(attr_constrain) == 2:
                            # two attributes
                            if examp_step < few_shot_num:
                                # first example (construct the candidate example with attribute1)
                                if chain_type == "action":
                                    random_idx = random.choice(pkl_data1)
                                else:
                                    [random_idx, cur_object_idx] = random.choice(pkl_data1)
                            else:
                                if chain_type == "action":
                                    random_idx = random.choice(pkl_data2)
                                else:
                                    [random_idx, cur_object_idx] = random.choice(pkl_data2)
                            
                        else:
                            # one attribute, first example
                            if chain_type == "action":
                                random_idx = random.choice(pkl_data1)
                            else:
                                [random_idx, cur_object_idx] = random.choice(pkl_data1)
                            
                    else:
                        if len(attr_constrain) == 2:
                            # two attributes
                            if accu_attr < 3:
                                if chain_type == "action":
                                    random_idx = random.choice(pkl_data1)
                                else:
                                    [random_idx, cur_object_idx] = random.choice(pkl_data1)
                                accu_attr += 1
                            elif accu_attr == 3:
                                if len(pkl_data1_data2) == 0:
                                    # re-initalize original image
                                    [cur_img_idx, cur_object_idx] = random.choice(pkl_data2)
                
                                    cur_obj = full_pkl[cur_img_idx]["objects"][cur_object_idx]
                                    cur_name_info = full_pkl[cur_img_idx]["name"]
                                    if chain_type == "affordance":
                                        cur_obj_attr = cur_obj['aff']
                                    elif chain_type == "attribute":
                                        cur_obj_attr = cur_obj['attr']
                                    
                                    cur_obj_box = cur_obj['box']
                                    cur_obj_name = cur_obj['obj']
                                    cur_image_path = os.path.join(file_dir[0][:data_dir_index],cur_name_info)
                                    cur_image = Image.open(cur_image_path)
                                    cur_image = cur_image.crop(cur_obj_box)
                                    cur_image = cur_image.resize((336,336)).convert('RGB')
                                    # select candidate positive image
                                    random.shuffle(pkl_data2)
                                    [random_idx, cur_object_idx] = random.choice(pkl_data2)
                                else:
                                    if chain_type == "action":
                                        random_idx = random.choice(pkl_data1_data2)
                                    else:
                                        [random_idx, cur_object_idx] = random.choice(pkl_data1_data2)
                                    
                                    
                                accu_attr += 1
                            elif accu_attr < 7:
                                if chain_type == "action":
                                    random_idx = random.choice(pkl_data2)
                                else:
                                    [random_idx, cur_object_idx] = random.choice(pkl_data2)
                                accu_attr += 1
                            elif accu_attr == 7:
                                if len(pkl_data1_data2) == 0:
                                    # re-initalize original image
                                    [cur_img_idx, cur_object_idx] = random.choice(pkl_data1)
                
                                    cur_obj = full_pkl[cur_img_idx]["objects"][cur_object_idx]
                                    cur_name_info = full_pkl[cur_img_idx]["name"]
                                    
                                    if chain_type == "affordance":
                                        cur_obj_attr = cur_obj['aff']
                                    elif chain_type == "attribute":
                                        cur_obj_attr = cur_obj['attr']
                                    cur_obj_box = cur_obj['box']
                                    cur_obj_name = cur_obj['obj']
                                    cur_image_path = os.path.join(file_dir[0][:data_dir_index],cur_name_info)
                                    cur_image = Image.open(cur_image_path)
                                    cur_image = cur_image.crop(cur_obj_box)
                                    cur_image = cur_image.resize((336,336)).convert('RGB')
                                    # select candidate positive image
                                    random.shuffle(pkl_data1)
                                    [random_idx, cur_object_idx] = random.choice(pkl_data1)
                                    
                                else:
                                    if chain_type == "action":
                                        random_idx = random.choice(pkl_data1_data2)
                                    else:
                                        [random_idx, cur_object_idx] = random.choice(pkl_data1_data2)
                                accu_attr = 0
                        else:
                            # one attribute 
                            if chain_type == "action":
                                random_idx = random.choice(pkl_data1)
                            else:
                                [random_idx, cur_object_idx] = random.choice(pkl_data1)
                    
                else:
                    if chain_type == "action":
                        random_idx = random.choice(negative_data)
                    else:
                        [random_idx, cur_object_idx] = random.choice(negative_data)
                
                if chain_type == "action":
                    cur_instance = full_pkl[random_idx]
                    random_obj_name, random_obj_chain, random_attr, random_img = get_pangea_img(file_dir[0], cur_instance,object_to_id)   
                        
                else:
                    random_info = full_pkl[random_idx]
                    random_img_info = random_info["objects"]
                    random_name_info = random_info["name"]
                    
                    random_obj_info = random_img_info[cur_object_idx]
                    
                    if chain_type == "affordance":
                        random_attr = random_obj_info['aff']
                    elif chain_type == "attribute":
                        random_attr = random_obj_info['attr']
                    random_box = random_obj_info['box']
                    random_obj_name = random_obj_info['obj']
                        
                    random_img_path = os.path.join(file_dir[0],random_name_info)    

                    random_img = Image.open(random_img_path)
                    random_img = random_img.crop(random_box)
                    random_img = random_img.resize((336,336)).convert('RGB')
                    
                co_attr = sorted(list(set(cur_obj_attr).intersection(set(random_attr))))
                co_attr = sorted(list(set(co_attr).intersection(set(attr_constrain_id))))
                if true_count < true_num:
                    assert len(co_attr) > 0
                
                if len(co_attr) > 0 and true_count < true_num:
                    image_banks.append(random_img)
                    attr_banks.append(random_attr)
                    true_gt_image = random_img
                    co_attr_bank.append(co_attr)
                    idx_banks.append(random_idx)
                    obj_name_banks.append(random_obj_name)
                    co_attr_label.append(1)
                    if chain_type == "affordance":
                        syne_step_log[f'gt_common_aff_true_{true_count}'] = [attr[int(attr_idx)] for attr_idx in co_attr]
                    elif chain_type == "attribute":
                        syne_step_log[f'gt_common_attr_true_{true_count}'] = [attr[int(attr_idx)] for attr_idx in co_attr]
                    elif chain_type == "action":
                        syne_step_log[f'gt_common_action_true_{true_count}'] = [attr[int(attr_idx)] for attr_idx in co_attr]
                    true_count += 1
                elif len(co_attr) == 0 and false_count < false_num:
                    image_banks.append(random_img)
                    attr_banks.append(random_attr)
                    idx_banks.append(random_idx)
                    co_attr_bank.append(co_attr)
                    obj_name_banks.append(random_obj_name)
                    co_attr_label.append(0)
                    false_count += 1
                
                else:
                    continue
                
                random_indices = random.sample(range(len(image_banks)),len(image_banks))
                image_banks = [image_banks[i] for i in random_indices]
                attr_banks = [attr_banks[i] for i in random_indices]
                idx_banks = [idx_banks[i] for i in random_indices]
                co_attr_bank = [co_attr_bank[i] for i in random_indices]
                obj_name_banks = [obj_name_banks[i] for i in random_indices]
                co_attr_label = [co_attr_label[i] for i in random_indices]
                
                if chain_type == "action":
                    question2_temp  = question2 + f"{selected_chain_clean}.\n"
                else:
                    question2_temp  = question2 + f"{selected_chain}.\n"
                
            if prompt_type in ["task_instruction", "task_instruction_tuned", "task_instruction_nlp_tuned", "task_instruction_nlp","task_instr_noimg", "semantic_instruct","task_instr_common_know","meta_common_know"]:
                # with instruction
                
                if prompt_type in ["task_instruction_nlp","task_instruction_nlp_tuned"]:
                    memory_base = f"Before this question, you have learnt that related pictures may have the following {chain_type}:\n"
                    for key, value in memory.items():
                        memory_base += f"{str(value)} have {str(key)} {chain_type},"
                    memory_base += ". Based on these knowledge, answer the follwing quesion:\n"
                else:
                    memory_base = f"Given the memory: {str(memory)} "
                
                qs = instruction_token + memory_base + instruction + question
                
                qs2 = instruction_token + instruction2 + question2_temp
            elif prompt_type in ["task_instruction_chainmem"]:
                memory_base = "Before this question, you have learnt that related pictures may have the following memory:\n"
                mem_chain = ''
                for key, value in memory.items():
                    for mem_idx,mem_item in enumerate(value):
                        if mem_idx == 0:
                            mem_chain += "->".join(mem_item)
                        else:
                            mem_chain += "->"
                            mem_chain += "->".join(mem_item[1:])
                    memory_base += mem_chain
                memory_base += ". Based on these knowledge, answer the follwing quesion:\n"
                
                qs = instruction_token + memory_base + instruction + question
                
                qs2 = instruction_token + instruction2 + question2_temp
                
            elif prompt_type in ["task_instruction_nomem"]:
                qs = instruction_token + instruction + question
                
                qs2 = instruction_token + instruction2 + question2_temp
            else:
                qs = question
                qs2 = question2_temp
            
            if each_steps < 5:
                logger.info(f"Prompt1: {qs}")
            
            if each_steps < 5:
                logger.info(f"Prompt2: {qs2}")
            
            if examp_step >= len(attr_constrain) * few_shot_num:
                input_images = [cur_image,] + image_banks
                
                # image_sizes = [x.size for x in input_images]
                
                if args.model_type == "llava-onevision":
                    response = QA_with_LLavaOneVision(llava_onevision_model,llava_onevision_tokenizer,llava_onevision_image_processor,qs,input_images) 
                elif args.model_type == "qwen2vl":
                    response = QA_with_Qwen2vl(qwen_model,qwen_processor,qs,input_images)
                elif args.model_type == "mplug3":
                    response = QA_with_mplug3(mplug3_model,mplug3_tokenizer,mplug3_processor,qs,input_images)
                elif args.model_type == "moe":
                    response = QA_with_MOE(qwen_model,qwen_processor,llava_onevision_model,llava_onevision_tokenizer,llava_onevision_image_processor,mplug3_model,mplug3_tokenizer,mplug3_processor,qs,input_images)
                true_pred_count = 0
                syne_object=False
                for index,label_index in enumerate(co_attr_label):
                    if label_index == 1:
                        syne_step_log["gt_label"] = label_dict[index][0]
                        syne_obj = []
                        for lad_item in label_dict[index]:
                            if lad_item in response:
                                syne_object = True
                                pred_inds = index
                                syne_obj.append(obj_name_banks[index])
                        # log
                        true_pred_count += 1
            else:
                syne_object = True
                pred_inds = co_attr_label.index(1)
            
            syne_attr = False
            if syne_object:
                gt_co_attr = co_attr_bank[pred_inds]
                
                if chain_type == "affordance":
                    syne_step_log['gt_co_aff'] = [attr[gt_attr_i] for gt_attr_i in gt_co_attr]
                elif chain_type == "attribute":
                    syne_step_log['gt_co_attr'] = [attr[gt_attr_i] for gt_attr_i in gt_co_attr]
                elif chain_type == "action":
                    syne_step_log['gt_co_action'] = [attr[gt_attr_i].split("-")[0] for gt_attr_i in gt_co_attr]
                
                input_images = [cur_image, image_banks[pred_inds]]
                if args.model_type == "llava-onevision":
                    response2 = QA_with_LLavaOneVision(llava_onevision_model,llava_onevision_tokenizer,llava_onevision_image_processor,qs2,input_images) 
                elif args.model_type == "qwen2vl":
                    response2 = QA_with_Qwen2vl(qwen_model,qwen_processor,qs2,input_images)
                elif args.model_type == "mplug3":
                    response2 = QA_with_mplug3(mplug3_model,mplug3_tokenizer,mplug3_processor,qs2,input_images)
                elif args.model_type == "moe":
                    response2 = QA_with_MOE(qwen_model,qwen_processor,llava_onevision_model,llava_onevision_tokenizer,llava_onevision_image_processor,mplug3_model,mplug3_tokenizer,mplug3_processor,qs2,input_images)
                
                true_pred_count = 0
                
                common_attr = []
                
                for attr_idx in gt_co_attr:
                    if attr[int(attr_idx)].lower() in response2.lower():
                        syne_attr = True
                        common_attr.append(attr[int(attr_idx)])
                
                generated_all_attr = []
                if chain_type == "action":
                    temp_chain_clean = selected_chain_clean
                else:
                    temp_chain_clean = selected_chain
                
                if prompt_type in ["task_instruction_chainmem"]:
                    for attr_name in temp_chain_clean:
                        if attr_name.lower() in response2.lower():
                            generated_all_attr.append(attr_name)
                    if len(memory.keys()) == 0:
                        memory["objects"] = [[cur_obj_name, ' '.join(generated_all_attr), obj_name_banks[pred_inds]]]
                    else:
                        memory["objects"].append([cur_obj_name, ' '.join(generated_all_attr), obj_name_banks[pred_inds]])
                    
                    if len(memory["objects"]) > 5:
                        memory["objects"] = memory["objects"][-5:]
                else:
                
                    for attr_name in temp_chain_clean:
                        if attr_name.lower() in response2.lower():
                            generated_all_attr.append(attr_name)
                            if attr_name in memory.keys():
                                if cur_obj_name not in memory[attr_name]:
                                    memory[attr_name].append(cur_obj_name)
                                    memory_weight[attr_name].append(1)
                                else:
                                    memory_weight[attr_name][memory[attr_name].index(cur_obj_name)] += 1
                                    
                                    
                                if obj_name_banks[pred_inds] not in memory[attr_name]:
                                    memory[attr_name].append(obj_name_banks[pred_inds])
                                    memory_weight[attr_name].append(1)
                                else:
                                    memory_weight[attr_name][memory[attr_name].index(obj_name_banks[pred_inds])] += 1

                            else:
                                if cur_obj_name == obj_name_banks[pred_inds]:
                                    memory[attr_name] = [cur_obj_name,]
                                    memory_weight[attr_name] = [2,]
                                else:
                                    memory[attr_name] = [cur_obj_name, obj_name_banks[pred_inds]]
                                    memory_weight[attr_name] = [1, 1]
                                
                    update_memory = {}    
                    update_memory_weight = {}
                    for key,value in memory_weight.items():
                        new_value = []
                        new_weight = []
                        for val_idx,value_item in enumerate(value):
                            if value_item - 0.2 > 0:
                                new_value.append(memory[key][val_idx])
                                new_weight.append(value_item - 0.2)
                        update_memory[key] = new_value
                        update_memory_weight[key] = new_weight
                    memory = update_memory
                    memory_weight = update_memory_weight
                        
                    # log
                    syne_step_log[f'pred_true_common_{chain_type}_{true_pred_count}'] = common_attr
                    syne_step_log[f'generated_all_common_{chain_type}_{true_pred_count}'] = generated_all_attr
            
                            
                syne_result = syne_object
                    
            else:
            
                syne_result = syne_object
            
                
            syne_step_log['syne_result'] = syne_result
            if examp_step >= len(attr_constrain) * few_shot_num:
                syne_step_log['response'] = response
            syne_step_log['response2'] = response2
            
            logger.info(f"Logs: {syne_step_log}")
            
            
            if examp_step < len(attr_constrain) * few_shot_num:
                examp_step += 1
                # example
                syne_result = True
                if examp_step < len(attr_constrain) * few_shot_num and len(attr_constrain) > 1:
                    if examp_step < few_shot_num:
            
                        cur_image = image_banks[pred_inds]
                        cur_obj_attr = attr_banks[pred_inds]
                        cur_obj_name = obj_name_banks[pred_inds]
                        cur_img_idx = idx_banks[pred_inds]
                    
                    elif examp_step == few_shot_num:
                        # two attribute second example
                        if len(pkl_data1_data2) > 0:
                            if chain_type == "action":
                                cur_img_idx = random.choice(pkl_data1_data2)
                            else:
                                [cur_img_idx, cur_object_idx] = random.choice(pkl_data1_data2)
                        else:
                            if chain_type == "action":
                                cur_img_idx = random.choice(pkl_data2)
                            else:
                                [cur_img_idx, cur_object_idx] = random.choice(pkl_data2)
                        
                        if chain_type == "action":
                            cur_instance = full_pkl[cur_img_idx]
                            cur_obj_name, cur_obj_chain, cur_obj_attr, cur_image = get_pangea_img(file_dir[0], cur_instance, object_to_id)
                            
                        else:
                        
                            cur_obj = full_pkl[cur_img_idx]["objects"][cur_object_idx]
                            cur_name_info = full_pkl[cur_img_idx]["name"]
                            if chain_type == "affordance":
                                cur_obj_attr = cur_obj['aff']
                            elif chain_type == "attribute":
                                cur_obj_attr = cur_obj['attr']
                            cur_obj_box = cur_obj['box']
                            cur_obj_name = cur_obj['obj']
                            cur_image_path = os.path.join(file_dir[0],cur_name_info)
                            cur_image = Image.open(cur_image_path)
                            cur_image = cur_image.crop(cur_obj_box)
                            cur_image = cur_image.resize((336,336)).convert('RGB')
                            
                    elif examp_step < len(attr_constrain) * few_shot_num - 1:
                        
                        cur_image = image_banks[pred_inds]
                        cur_obj_attr = attr_banks[pred_inds]
                        cur_obj_name = obj_name_banks[pred_inds]
                        cur_img_idx = idx_banks[pred_inds]
                    else:
                        
                        if len(pkl_data1_data2) > 0:
                            if chain_type == "action":
                                cur_img_idx = random.choice(pkl_data1_data2)
                            else:
                                [cur_img_idx, cur_object_idx] = random.choice(pkl_data1_data2)
                        else:
                            if chain_type == "action":
                                cur_img_idx = random.choice(pkl_data2)
                            else:
                                [cur_img_idx, cur_object_idx] = random.choice(pkl_data1)
                        
                        if chain_type == "action":
                            cur_instance = full_pkl[cur_img_idx]
                            cur_obj_name, cur_obj_chain, cur_obj_attr, cur_image = get_pangea_img(file_dir[0], cur_instance, object_to_id)
                            
                        else:
                        
                            # two attribute second example
                            if len(pkl_data1_data2) > 0:
                                [cur_img_idx, cur_object_idx] = random.choice(pkl_data1_data2)
                            else:
                                [cur_img_idx, cur_object_idx] = random.choice(pkl_data1)
                            
                            cur_obj = full_pkl[cur_img_idx]["objects"][cur_object_idx]
                            cur_name_info = full_pkl[cur_img_idx]["name"]
                            if chain_type == "affordance":
                                cur_obj_attr = cur_obj['aff']
                            elif chain_type == "attribute":
                                cur_obj_attr = cur_obj['attr']
                            cur_obj_box = cur_obj['box']
                            cur_obj_name = cur_obj['obj']
                            
                            
                            cur_image_path = os.path.join(file_dir[0],cur_name_info)
                            cur_image = Image.open(cur_image_path)
                            cur_image = cur_image.crop(cur_obj_box)
                            cur_image = cur_image.resize((336,336)).convert('RGB')
                            
                else:
                    if chain_type == "action":
                        cur_img_idx = random.choice(pkl_data1)
                        cur_instance = full_pkl[cur_img_idx]
                        cur_obj_name, cur_obj_chain, cur_obj_attr, cur_image = get_pangea_img(file_dir[0], cur_instance, object_to_id)
                        
                    else:
                        [cur_img_idx, cur_object_idx] = random.choice(pkl_data1)
                    
                        cur_obj = full_pkl[cur_img_idx]["objects"][cur_object_idx]
                        cur_name_info = full_pkl[cur_img_idx]["name"]
                        if chain_type == "aff":
                            cur_obj_attr = cur_obj['aff']
                        elif chain_type == "attr":
                            cur_obj_attr = cur_obj['attr']
                            
                        cur_obj_box = cur_obj['box']
                        cur_obj_name = cur_obj['obj']
                        cur_image_path = os.path.join(file_dir[0],cur_name_info)
                        cur_image = Image.open(cur_image_path)
                        cur_image = cur_image.crop(cur_obj_box)
                        cur_image = cur_image.resize((336,336)).convert('RGB')
                
            elif syne_result:
                cur_image = image_banks[pred_inds]
                cur_obj_attr = attr_banks[pred_inds]
                cur_obj_name = obj_name_banks[pred_inds]
                cur_img_idx = idx_banks[pred_inds]
                accu_step += 1
                
            syne_proc.append(syne_step_log)
                
        logger.info(f"Step: {each_steps}, Accumulation step: {accu_step}, Syne Chain: {syne_log}")

        accu_steps_list.append(accu_step)
        
        if each_steps % 100 == 0 and each_steps > 0:
            
            eval_result = compute_metrics(accu_steps_list)
            
            for key,value in eval_result.items():
                logger.info(f"{key}: {value}")
    

    logger.info(f"Eval finished!")
    eval_result = compute_metrics(accu_steps_list)
    
    for key,value in eval_result.items():
        logger.info(f"{key}: {value}")
    

if __name__ == "__main__":
    main()