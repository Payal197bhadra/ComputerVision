import random, os
import numpy as np
from PIL import Image
path = r"C:\Datasets\AIDER\traffic_incident"
# random_filename = random.choice([
#     x for x in os.listdir(path)
#     if os.path.isfile(os.path.join(path, x))
# ])

files= [
    x for x in os.listdir(path)
    if os.path.isfile(os.path.join(path, x))
]
random_files = np.random.choice(files, int(len(files)*.19))
print(random_files)
print(len(random_files))
import shutil
for i in random_files:
  original="C:/Datasets/AIDER/traffic_incident/"+i
  store_path = "C:/Datasets/AIDER/multiclass/test/traffic_incident/"+i
  shutil.move(original, store_path)


# import numpy as np
# import os
# path = r"C:\Datasets\AIDER\normal"
# # list all files in dir
# files = [f for f in os.listdir(path)
#          if os.path.isfile(os.path.join(path, f)]
#
# # select 0.1 of the files randomly
# random_files = np.random.choice(files, int(len(files)*.1))

# print(random_files)