import os
import json
import torch
import random
import pickle
import logging
import torchvision

import numpy as np 
from typing import List, Dict,Any,NewType
from PIL import Image
from dataclasses import dataclass
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from transformers.models.clip import CLIPTokenizer
InputDataClass = NewType("InputDataClass",Any)


class OCLDataset(Dataset):
    def __init__(self,data_root,mode, attr_constraint:str="no",data_type="ocl_attr_data"):
        super(OCLDataset,self).__init__()
        self.logger = logging.getLogger(f"Dataset OCL")
        self.data_root = data_root
        self.mode = mode
        self.data_type = data_type
        self.attr_constraint = attr_constraint.split(",")
        
        anno_dir = os.path.join(data_root,"OCL_data","data","resources")
        
        def load_pickle_and_assign_split(pkl_dir,split):
            
            pkl_path = os.path.join(pkl_dir, f"OCL_annot_{split}.pkl")
            
            if self.data_type == "ocl_attr_data":
                preprocessed_pkl_path = os.path.join(pkl_dir, "OCL_selected_test_attribute_refined_new_1.pkl")
            elif self.data_type == "ocl_aff_data":
                preprocessed_pkl_path = os.path.join(pkl_dir, "OCL_selected_test_affordance_refined_new_1_v2.pkl")
            with open(pkl_path, 'rb') as fp:
                pkl = pickle.load(fp)
            for x in pkl:
                x['split'] = split
            
            with open(preprocessed_pkl_path, 'rb') as fp:
                preprocessed_pkl = pickle.load(fp)
                
            return pkl,preprocessed_pkl
            
        if mode == "test":
            example_num = 0
        
        if self.mode == "train":
            self.pkl_data,self.preprocessed_pkl_data = load_pickle_and_assign_split(anno_dir,"train")
        elif self.mode == "val":
            self.pkl_data,self.preprocessed_pkl_data = load_pickle_and_assign_split(anno_dir,"test")
        else:
            self.pkl_data, self.preprocessed_pkl_data = load_pickle_and_assign_split(anno_dir,"test")
        
        self.instance_indices = [(i,j) for i,img in enumerate(self.pkl_data) for j in range(len(img['objects']))]
        self.logger.info(f"{len(self.instance_indices)} instances")
        
        def load_class_json(name):
            with open(os.path.join(anno_dir,f"OCL_class_{name}.json"),"r") as fp:
                return json.load(fp) 
        
        
        self.attrs = load_class_json("attribute")
        self.objs = load_class_json("object")
        aff_dict = load_class_json("affordance")
        
        self.affs = []
        for aff_item in aff_dict:
            if aff_item["word"][0] not in self.affs:
                self.affs.append(aff_item["word"][0])
            else:
                if len(aff_item["word"]) > 1:
                    random_aff = random.choice(aff_item["word"])
                    while random_aff in self.affs and random_aff == aff_item["word"][0]:
                        random_aff = random.choice(aff_item["word"])
                    self.affs.append(random_aff)
                else:
                    if len(aff_dict[self.affs.index(aff_item["word"][0])]["word"]) > 1:
                        random_aff = random.choice(aff_dict[self.affs.index(aff_item["word"][0])]["word"])
                        while random_aff in self.affs and random_aff == aff_item["word"][0]:
                            random_aff = random.choice(aff_dict[self.affs.index(aff_item["word"][0])]["word"])
                            
                        self.affs[self.affs.index(aff_item["word"][0])] = random_aff
                        self.affs.append(aff_item["word"][0])
                    else:
                        self.affs.append(aff_item["word"][0] + '_1')
        
        updated_affs = []
        for aff_item in self.affs:
            if aff_item == "write":
                updated_affs.append("imprint")
            else:
                updated_affs.append(aff_item)
            
        self.affs = updated_affs
                    
        
        self.obj2id = {x: i for i, x in enumerate(self.objs)}
        
        if len(self.attr_constraint) == 2:
            self.attr1_pkl = self.preprocessed_pkl_data["selected_paired_pkl"][self.attr_constraint[0] + "-" + self.attr_constraint[1]][self.attr_constraint[0]]
            self.attr2_pkl = self.preprocessed_pkl_data["selected_paired_pkl"][self.attr_constraint[0] + "-" + self.attr_constraint[1]][self.attr_constraint[1]]
            self.attr1_attr2_pkl = self.preprocessed_pkl_data["selected_paired_pkl"][self.attr_constraint[0] + "-" + self.attr_constraint[1]][self.attr_constraint[0] + "-" + self.attr_constraint[1]]
            self.negative_pkl = self.preprocessed_pkl_data["negative_pkl"]
            num_samples = min(len(self.attr1_pkl),len(self.attr2_pkl))
        
        elif len(self.attr_constraint) == 1:
            self.attr1_pkl = self.preprocessed_pkl_data["selected_individual_pkl"][self.attr_constraint[0]]
            self.negative_pkl = self.preprocessed_pkl_data["negative_pkl"]
            num_samples = len(self.attr1_pkl)
        
        with open(os.path.join(anno_dir,"category_aff_matrix.json"),"r") as fp:
            aff_matrix_file = json.load(fp)
            assert self.objs == aff_matrix_file["objs"]
            self.aff_matrix = np.array(aff_matrix_file["aff_matrix"]) # [380, 170]
            
        self.num_aff = self.aff_matrix.shape[1] # 380
        self.num_attr = len(self.attrs)
        self.num_obj = len(self.objs)
        
        self.logger.info("#obj %d, #attr %d, #aff %d" % (
            self.num_obj, self.num_attr, self.num_aff))
        
        
        self.length = max(num_samples,1000)
        
    def __len__(self):
        return self.length

    def __getitem__(self,index):
        info = self.pkl_data[index]
        
        file_path = os.path.join(self.data_root,"OCL_data","data")
        
        if (self.attr_constraint[0] != "no" and self.mode == "test"): 
            if len(self.attr_constraint) == 2:
                combined_pkl = [self.pkl_data,self.attr1_pkl, self.attr2_pkl, self.attr1_attr2_pkl,self.negative_pkl]
            else:
                combined_pkl = [self.pkl_data,self.attr1_pkl,self.negative_pkl]
        else:
            combined_pkl = self.pkl_data
        if self.data_type == "ocl_attr_data":
            sample = {"idx":index,
                    "file_dir":file_path,
                    "pkl_data":combined_pkl,
                    "attr":self.attrs,
                    "labels":torch.tensor(index)}
            return sample

        elif self.data_type == "ocl_aff_data":
            sample = {"idx":index,
                    "file_dir":file_path,
                    "pkl_data":combined_pkl,
                    "attr":self.affs,
                    "labels":torch.tensor(index)}
            return sample
        
def OCL_data_collate(features:List[InputDataClass]):
    batch = {}
    batch["idx"] = [item["idx"] for item in features]
    batch["file_dir"] = [item["file_dir"] for item in features]
    batch["pkl_data"] = features[0]["pkl_data"]
    batch["attr"] = features[0]["attr"]
    batch["labels"] = torch.stack([item["labels"] for item in features],dim=0)
    
    return batch