import os
import pickle
from collections import Counter
image_path = "t_set"
file_dict = pickle.load(open("file_dict", "rb"))
file_dict = dict(file_dict)


cnt = Counter()
for v in file_dict.values():
    cnt[v] += 1

new_dict = dict()

for k, v in file_dict.items():
    assert cnt[v] >= 0
    new_k = "{}/{}_{:d}.jpg".format(image_path, v, cnt[v])
    os.rename(k, new_k)
    new_dict.update({new_k:v})
    cnt[v] -= 1

pickle.dump(new_dict, open("renew_file_dict", "wb"))
