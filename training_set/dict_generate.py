import os
from collections import Counter
import pickle


img_dir = "merge_data/"
file_list = os.listdir(img_dir)
cnt = Counter()
file_dict = dict()


for file in file_list:
    file_label = file[:4]
    cnt[file_label] += 1
    new_file_name = "{0}{1}@{2}.jpg".format(img_dir, file_label, cnt[file_label])
    print new_file_name
    os.rename("{0}{1}".format(img_dir, file), new_file_name)
    file_dict.update({new_file_name:file_label})
    #print "{0}{1}_{2}.jpg".format(img_dir, file_label, cnt[file_label])

	
with open("nofile_dict", "wb") as f:
	pickle.dump(file_dict, f)
	f.close()
