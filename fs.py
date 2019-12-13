import os
from os.path import join
import shutil


def listimg(path):
    return [item for item in os.listdir(path)\
        if item.lower().endswith(
            ('.jpg','.jpeg', '.png', '.bmp', '.jp2', '.tif', '.tiff'))]

def ensuredir(*args, file_name=None):
    path = join(*args)
    if not os.path.exists(path): 
        os.makedirs(path)
    elif not os.path.isdir(path):
        raise FileExistsError
    if file_name is not None:
        path = join(path, file_name)
    return path

def removeall(dir):
    if not os.path.exists(dir):return
    for file in os.listdir(dir):
        file = os.path.join(dir,file)
        if os.path.isdir(file):
            removeall(file)
        else:
            os.remove(file)
    shutil.rmtree(dir)