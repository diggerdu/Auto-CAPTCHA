#from __future__ import print_function
from PIL import Image 
import pickle
import numpy as np
import string



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

    im = np.asarray(Image.open("test.img").convert('L'))
    im = im - 120
    im = im / (255.000 - 120)
    for start in range(3, 40, 12):
        tmp = im[:,start:start+20]
        image_data.append(tmp)

image_data = np.asarray(image_data)

np.save(image_data_path, image_data)
