from PIL import Image
import os
from collections import Counter

dir = "test/"

file_list = os.listdir(dir)
file_list = map(lambda name:dir+name, file_list)

cnt = Counter()

for file in file_list:
    print file
    Im = Image.open(file)
    Im.show()
    right_label = raw_input('please input what you see: ')
    print right_label
    cnt[right_label] += 1
    os.rename(file, dir + right_label +'_'+str(cnt[right_label]) + ".jpg")
    os.system("pkill display")
