#from __future__ import print_function
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
    idx += 1

for char in string.lowercase:
    char_table[char] = idx
    idx += 1

for i in range(10):
    char_table[str(i)] = idx
    idx += 1

print (len(char_table))
with open("file_dict","rb") as f:
    file_dict = pickle.load(f)



target_data = list()
image_data = list()

for key, value in file_dict:
    target = map(lambda char:char_table[char], value)
    target_data.extend(target)
    im = np.asarray(Image.open(key).convert('L')) / 255.000
    for start in range(3, 40, 12):
        tmp = im[:,start:start+20]
        image_data.append(tmp)

target_data = np.asarray(target_data)
image_data = np.asarray(image_data)

np.save(image_data_path, image_data)
np.save(target_data_path, target_data)
