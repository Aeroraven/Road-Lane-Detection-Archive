import shutil
import os


def fcpy(srcfile, dstfile, remove=True):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % srcfile)
        raise Exception("file not found")
    else:
        fpath = os.path.dirname(srcfile)
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        shutil.copyfile(srcfile, dstfile)
        if remove:
            os.remove(srcfile)


def file_readlines(src):
    ret = []
    with open(src, "r") as f:
        ret = f.readlines()
    return ret
