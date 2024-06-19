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
        self.data = read_pkl_file("./../../data/OCL_test_data.pkl")
        self.path = path
        self.attr_name = ['wooden', 'metal', 'flying', 'ripe', 'fresh', 'natural', 'cooked', 'painted', 'rusty', 'furry']
        self.available_attr = self.attr_name
        self.attr2index = {}
        for idx, attr in enumerate(self.attr_name):
            self.attr2index[attr] = idx
        self.used = set()
        self.attr2item_index = {}
        self.attr2category = {}
        for attr in self.attr_name:
            self.attr2item_index[attr] = []
        for idx, item in enumerate(self.data):
            for obj in item["objects"]:
                for att in obj["attr"]:
                    if self.attr_name[att] in self.available_attr:
                        #print(obj)
                        self.attr2item_index[self.attr_name[att]].append({'image_id':idx, 'object': obj})
    
    # {'image_id':idx, 'object': obj}
    # to a PIL image that have been resized to 200, 200
    def process_img(self, img):
        name = self.data[img["image_id"]]['name']
        name = os.path.basename(name)
        image = Image.open(os.path.join(self.path, name))
        box = img["object"]['box']
        image = image.crop(box).resize((200, 200))
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
        
        query = random.choice(self.attr2item_index[attr])
        
        img1 = None
        while img1 == None:
            rand_img = random.choice(self.attr2item_index[attr])
            if rand_img["image_id"] != query["image_id"]:
                img1 = rand_img
                self.used.add(img1["image_id"])
                
        img2 = None
        while img2 == None:
            wrong_attr = self.get_attr(attr)
            #print(f"wrong:{wrong_attr}")
            rand_img = random.choice(self.attr2item_index[wrong_attr])
            if rand_img["image_id"] != query["image_id"] and rand_img["image_id"] != img1["image_id"] and len(set(rand_img["object"]['attr']).intersection(set(query["object"]['attr']))) == 0 and rand_img["image_id"] not in self.used:
                img2 = rand_img
                self.used.add(img2["image_id"])
        self.query = img1
        return [query, img1, img2]

    def get_next(self, attr):
        img1 = None
        while img1 == None:
            rand_img = random.choice(self.attr2item_index[attr])
            if rand_img["image_id"] != self.query["image_id"] and rand_img["image_id"] not in self.used:
                img1 = rand_img
                self.used.add(img1["image_id"])
                
        img2 = None
        while img2 == None:
            wrong_attr = self.get_attr(attr)
            #print(f"wrong:{wrong_attr}")
            rand_img = random.choice(self.attr2item_index[wrong_attr])
            if rand_img["image_id"] != self.query["image_id"] and rand_img["image_id"] != img1["image_id"] and len(set(rand_img["object"]['attr']).intersection(set(self.query["object"]['attr']))) == 0 and rand_img["image_id"] not in self.used:
                img2 = rand_img
                self.used.add(img2["image_id"])
                
        self.query = img1 
        return [img1, img2]