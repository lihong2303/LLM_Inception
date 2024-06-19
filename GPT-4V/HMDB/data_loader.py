import pickle
import random
import os
import warnings
from PIL import Image

warnings.filterwarnings('ignore')

def read_pkl_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

class data_loader(object):
    
    def __init__(self, path):
        self.data = read_pkl_file("./../../data/HMDB.pkl")
        self.path = path
        self.attr_name = ['brush_hair','clap', 'dive', 'shake_hands','hug' ,'sit','smoke','eat']
        self.attr2idx = {}
        for attr in self.attr_name:
            
            self.attr2idx[attr] = []
        for idx,item in enumerate(self.data):
            if item["label"] != "climb" and item["label"] != "ride_horse":
                self.attr2idx[item['label']].append(idx)
        
    # {'image_id':idx, 'object': obj}
    # to a PIL image that have been resized to 200, 200
    def process_img(self, img):
        name = self.data[img]['name']
        image = Image.open(os.path.join(self.path, name)).resize((200, 200))
        return image
    
    def print_attr(self, attr):
        attrr = []
        for i in attr:
            attrr.append(self.attr_name[i])
        print(attrr)
    
    def get_attr(self, attr):
        res = None
        while res == None:
            rand_attr = random.choice(self.attr_name)
            if rand_attr != attr:
                res = rand_attr
        #print(res)
        return res
    
    # return [query, img1, img2]
    # img1 is always the correct one
    # attr is the shared attr in query and img1
    # this create an example as a shot should be different from attr and in the same category
    # moreover, query and img2 should have no shared attr
    def get_data(self, attr = None):
        
        query = random.choice(self.attr2idx[attr])
        
        img1 = None
        while img1 == None:
            rand_img = random.choice(self.attr2idx[attr])
            if rand_img != query:
                img1 = rand_img
                
        img2 = None
        while img2 == None:
            wrong_attr = self.get_attr(attr)
            rand_img = random.choice(self.attr2idx[wrong_attr])
            if rand_img != query and rand_img != img1:
                img2 = rand_img
        self.query = img1
        return [query, img1, img2]

    def get_next(self, attr):
        img1 = None
        while img1 == None:
            rand_img = random.choice(self.attr2idx[attr])
            if rand_img != self.query:
                img1 = rand_img
                
        img2 = None
        while img2 == None:
            wrong_attr = self.get_attr(attr)
            rand_img = random.choice(self.attr2idx[wrong_attr])
            if rand_img != self.query and rand_img != img1:
                img2 = rand_img
                
        self.query = img1 
        return [img1, img2]