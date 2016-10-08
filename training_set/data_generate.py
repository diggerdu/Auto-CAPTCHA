from __future__ import print_function
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


image_data = list()
label_data = list()

for key, value in file_dict:
    im = np.asarray(Image.open(key).convert('L'))
    s = im.shape
    im = im.reshape((s[1], s[0])) / 255.00000
    image_data.append(im) 
    label_data.append(map(lambda char:char_table[char], value))

print (label_data[0])

image_data = np.asarray(image_data)
label_data = np.asarray(label_data)
np.save("image_data", image_data)
np.save("label_data", label_data)

print(image_data.shape)
print(label_data.shape)
