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
    
    def __init__(self, path, paired, type):
        if type != 'action':
            self.data = read_pkl_file("./../data/OCL_annot_test.pkl")
            self.cross = read_pkl_file(f"./../data/OCL_selected_test_{type}_refined_new_1.pkl")
        else:
            self.cross = read_pkl_file(f"./../data/pangea_test_refined_new.pkl")
            self.data = read_pkl_file(f"./../data/B123_test_KIN-FULL_with_node.pkl")
        self.paired_attr = self.cross['selected_paired_pkl'][paired].keys()
        self.paired = paired
        self.paired_data = self.cross['selected_paired_pkl'][paired]
        self.negative_data = self.cross['negative_pkl']
        self.path = path
        self.now_attr = None
        self.iteration_cnt = 0
        self.query = None
        self.type = type
        
    # [img_idx, obj_idx]
    # to a PIL image that have been resized to 336, 336
    def process_img(self, img):
        if self.type == 'action':
            name = self.data[img][1]
        else:
            name = self.data[img[0]]['name']
        # name = os.path.basename(name)
        # print(name)
        image = Image.open(os.path.join(self.path, name))
        if image.mode == 'RGBA' or image.mode == 'P':
            # Convert the image to RGB
            image = image.convert('RGB')
        if self.type == 'action':
            image = image.resize((336, 336))
        else:
            box = self.data[img[0]]['objects'][img[1]]['box']
            image = image.crop(box).resize((336, 336))
        return image
    
    def get_new_example(self, attr):
        img1 = None
        while img1 == None:
            rand_img = random.choice(self.attr2item_index[attr])
            img1 = rand_img
        img1 = self.process_img(img1)
        return img1
            
    # return [query, img1, img2]
    # img1 is always the correct one
    # attr is the shared attr in query and img1
    # this create an example as a shot should be different from attr and in the same category
    # moreover, img2 is from negative data, promising that img2 is distingushable enough from img1
    def get_data(self, attr = None):
        
        query = random.choice(self.paired_data[attr])

        img1 = None
        while img1 == None:
            rand_img = random.choice(self.paired_data[attr])
            if self.type != 'action' and rand_img[0] != query[0]:
                img1 = rand_img
            if self.type == 'action' and rand_img != query:
                img1 = rand_img
                
        img2 = None
        while img2 == None:
            rand_img = random.choice(self.negative_data)
            if self.type != 'action' and rand_img[0] != query[0]:
                img2 = rand_img
            if self.type == 'action' and rand_img != query:
                img2 = rand_img

        self.query = img1
        self.now_attr = attr
        return [query, img1, img2]

    def get_next(self, attr):
        self.iteration_cnt += 1
        img1 = None
        if self.iteration_cnt % 5 == 4 and len(self.cross) > 0:
            rand_cross = random.choice(self.paired_data[self.paired])
            img1 = rand_cross
            
        while img1 == None:
            rand_img = random.choice(self.paired_data[attr])
            if self.type != 'action' and rand_img[0] != self.query[0]:
                img1 = rand_img
            if self.type == 'action' and rand_img != self.query:
                img1 = rand_img
                
        img2 = None
        while img2 == None:
            rand_img = random.choice(self.negative_data)
            if self.type != 'action' and rand_img[0] != self.query[0]:
                img2 = rand_img
            if self.type == 'action' and rand_img != self.query:
                img2 = rand_img
        self.query = img1 
        
        return [img1, img2]