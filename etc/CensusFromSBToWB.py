import os
import shutil
import zipfile
import lzma as xz
import subprocess
import py7zr

remove_existing_files = True
files_unzipped = True
db_root = r"C:\work\water_use\mldataset"
ppath = r"ml\training\features"
download_folder = r"C:\work\water_use\census_wsa\census_3_3_22\WSA_places_yearly"
file_list = os.listdir(download_folder)
type_of_census = "census_places" # options are : census_bg and census_places

for file in file_list:
    print(file)
    huc2 = file.split(".")[0].replace("huc", "")
    huc2 = int(huc2)
    fn = os.path.join(db_root, ppath)
    fn = os.path.join(fn, str(huc2))

    if not(os.path.isdir(fn)):
        os.mkdir(fn)

    fn = os.path.join(fn, type_of_census)
    if remove_existing_files:
        if os.path.isdir(fn):
            shutil.rmtree(fn)

    if not(os.path.isdir(fn)):
        os.mkdir(fn)

    src = os.path.join(download_folder, file)
    dst = os.path.join(fn, file)

    if files_unzipped:
        shutil.copytree(src, dst)
        unzipped_folder = dst
    else:
        try:
            shutil.copy(src = src, dst = dst  )
        except:
            print(" {} exist".format(dst))

        with py7zr.SevenZipFile(dst, 'r') as archive:
            archive.extractall(path=os.path.dirname(dst))
        unzipped_folder = os.path.join(os.path.dirname(dst),  os.path.splitext(file)[0])

    unzList = os.listdir(unzipped_folder)
    for ff in unzList:
        ssrc  = os.path.join(unzipped_folder, ff)
        if not("_yearly" in ssrc):
            continue
        print(ff)
        ddst  = os.path.join(os.path.dirname(dst))
        if os.path.isfile(os.path.join(ddst,ff)):
            os.remove(os.path.join(ddst,ff))
        shutil.move(ssrc, ddst)
    #os.remove(dst)
    shutil.rmtree(unzipped_folder)
    ccc = 1


