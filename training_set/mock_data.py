from __future__ import print_function
from PIL import Image
import pickle
import numpy as np
import string


image_data_path = "image_data/"
target_data_path = "target_data/"

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


cnt = 0
for key, value in file_dict:
    im = np.asarray(Image.open(key).convert('L')) / 255.00000
    im = im.reshape((60,22))
    im = im.reshape((6,220))
    im = im.transpose()
    print (im.shape)
    np.save(image_data_path + str(cnt),im)
    target = np.asarray(map(lambda char:char_table[char], value))
    np.save(target_data_path + str(cnt), target)
    cnt += 1
