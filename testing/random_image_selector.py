import shutil, random, os
dirpath = 'data/images/imgs_part_3'
destDirectory = 'testimages'

filenames = random.sample(os.listdir(dirpath), 100)
for fname in filenames:
    srcpath = os.path.join(dirpath, fname)
    shutil.copy(srcpath, destDirectory)
