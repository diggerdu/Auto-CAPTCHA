from __future__ import print_function
from PIL import Image 
import pickle
import numpy as np
import string


image_data_path = "image_data"
target_data_path = "target_data"

char_table = dict()
idx = 0;

for char in string.uppercase:
    char_table[char] = idx
    char_table[char.lower()] = idx
    idx += 1


for i in range(10):
    char_table[str(i)] = idx
    idx += 1

print (len(char_table))
print (char_table)
with open("nofile_dict","rb") as f:
    file_dict = pickle.load(f)



target_data = list()
image_data = list()

for key, value in file_dict.items():
    tmp_extend = list()
    target = map(lambda char:char_table[char], value)

    target_data.append(target)
    im = np.asarray(Image.open(key).convert('L'))
    im = im - 120
    im = im / (255.000 - 120)
    for start in range(3, 40, 12):
        tmp_single = im[1:21,start:start+20]
        tmp_extend.append(tmp_single)
    tmp_extend = np.asarray(tmp_extend)
    #print (tmp_extend.shape)
    image_data.append(np.reshape(tmp_extend, (80,20)))
        

target_data = np.asarray(target_data)
target_data = np.expand_dims(target_data, axis=-1)
target_data = np.asarray(np.arange(36) == target_data, dtype='f')
target_data.shape
target_data.shape
image_data = np.asarray(image_data)

np.save(image_data_path, image_data)
np.save(target_data_path, target_data)
