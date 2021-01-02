import os
from os.path import basename
import py7zr
from zipfile import ZipFile

def from_zip(src_file, dst_dir, post_process = None):
    try:
        with ZipFile(src_file, 'r') as zip_ref:
            zip_ref.extractall(dst_dir)
    except:
        with py7zr.SevenZipFile(dst_dir, 'r') as archive:
            archive.extractall(path=os.path.dirname(dst_dir))
    if post_process:
        post_process(src_file, dst_dir)


def to_zip(target_dir, outzip_name, filter):
    with ZipFile(outzip_name, 'w') as zipObj:
        for folderName, subfolders, filenames in os.walk(target_dir):
            for filename in filenames:
                if filter(filename):
                    filePath = os.path.join(folderName, filename)
                    zipObj.write(filePath, basename(filePath))


if __name__ == "__main__":
    pass