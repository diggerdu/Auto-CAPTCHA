from PIL import Image
import numpy as np

a = Image.open("0.jpg").convert('L')
im = np.asarray(Image.open("0.jpg").convert('L'))
print im.shape
a.save("hate.jpg")
