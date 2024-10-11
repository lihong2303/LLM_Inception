
import os
import logging
import pickle
import torch
from typing import List, Dict,Any,NewType
from torch.utils.data import Dataset
InputDataClass = NewType("InputDataClass",Any)

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

class PangeaDataset(Dataset):
    def __init__(self, data_root, mode, chain_constraint:str="no"):
        super(PangeaDataset,self).__init__()
        
        self.logger = logging.getLogger(f"Dataset Pangea")
        self.data_root = data_root
        self.mode = mode
        
        self.chain_constraint = chain_constraint.split(",")
        self.chain_constraint = [map_dict[chain] for chain in self.chain_constraint]
        
        annotation_dir = os.path.join(data_root, "pangea")
        
        def load_pickle_and_assign_split(pkl_dir, split):
            if split == "test":
                pkl_path = os.path.join(pkl_dir, f"B123_{split}_KIN-FULL_with_node.pkl")
            else:
                pkl_path = os.path.join(pkl_dir, "B123_test_KIN-FULL_with_node.pkl")
            
            preprocessed_pkl_path = os.path.join(pkl_dir, "pangea_test_refined_new.pkl")
            
            with open(pkl_path, 'rb') as fp:
                pkl = pickle.load(fp)
                
            with open(preprocessed_pkl_path, 'rb') as fp:
                preprocessed_pkl = pickle.load(fp)
                
            return pkl,preprocessed_pkl

        
        if self.mode == "train":
            self.pkl_data,self.preprocessed_pkl_data = load_pickle_and_assign_split(annotation_dir,"train")
            
        elif self.mode == "test":
            self.pkl_data,self.preprocessed_pkl_data = load_pickle_and_assign_split(annotation_dir,"test")
            
        
        with open(os.path.join(annotation_dir, "mapping_node_index.pkl"),"rb") as fb:
            mapping_node_index = pickle.load(fb)
        
        with open(os.path.join(annotation_dir, "verbnet_topology_898.pkl"),"rb") as fb:
            verbnet_topology = pickle.load(fb)
        objects = verbnet_topology["objects"]
        
        self.objects_290 =  objects[mapping_node_index]
        
        if len(self.chain_constraint) == 2:
            self.chain1_pkl = self.preprocessed_pkl_data["selected_paired_pkl"][self.chain_constraint[0] + '_' + self.chain_constraint[1]][self.chain_constraint[0]]
            self.chain2_pkl = self.preprocessed_pkl_data["selected_paired_pkl"][self.chain_constraint[0] + '_' + self.chain_constraint[1]][self.chain_constraint[1]]
            self.chain1_chain2_pkl = self.preprocessed_pkl_data["selected_paired_pkl"][self.chain_constraint[0] + '_' + self.chain_constraint[1]][self.chain_constraint[0] + '_' + self.chain_constraint[1]]
            self.negative_pkl = self.preprocessed_pkl_data["negative_pkl"]
            num_samples = min(len(self.chain1_pkl),len(self.chain2_pkl))
        
        elif len(self.chain_constraint) == 1:
            self.chain1_pkl = self.preprocessed_pkl_data["selected_individual_pkl"][self.chain_constraint[0]]
            self.negative_pkl = self.preprocessed_pkl_data["negative_pkl"]
            num_samples = len(self.chain1_pkl)
        
        self.length = max(num_samples, 1000)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        
        file_path = os.path.join(self.data_root, "pangea","pangea_new")
        if (self.chain_constraint[0] != "no" and self.mode == "test"): 
            if len(self.chain_constraint) == 2:
                combined_pkl = [self.pkl_data,self.chain1_pkl, self.chain2_pkl, self.chain1_chain2_pkl,self.negative_pkl]
            else:
                combined_pkl = [self.pkl_data,self.chain1_pkl,self.negative_pkl]
        else:
            combined_pkl = self.pkl_data
        
        sample = {"idx":index,
                "file_dir":file_path,
                "pkl_data":combined_pkl,
                "attr":list(self.objects_290),
                "labels":torch.tensor(index)}
        return sample
        
def Pangea_data_collate(features:List[InputDataClass]):
    batch = {}
    batch["idx"] = [item["idx"] for item in features]
    batch["file_dir"] = [item["file_dir"] for item in features]
    batch["pkl_data"] = features[0]["pkl_data"]
    batch["attr"] = features[0]["attr"]
    batch["labels"] = torch.stack([item["labels"] for item in features],dim=0)
    
    return batch
