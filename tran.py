import os
import shutil
import tqdm

dire = './douy_data_resize/'
for curdir,subdirs,files in os.walk(dire):
        for i,file in enumerate(files):
            path = os.path.join(curdir,file)
            shutil.copy(path,'./all_douy_resize/{}'.format(file))
