# ScienceBase Synchronization Code
The python code in this folder is for synchronizing local files and folders with ScienceBase.
## sbsync.py
sbsync.py contains a simple python library for synchronizing a ScienceBase archive with a local structure of files and folders.  It uses the ScienceBasePy libarary to communicate with ScienceBase.  All calls to ScienceBasePy are made through the SBAccess class, which encapsulates certain ScienceBasePy operations adding error/retry code to the operations.  By default SBAccess tries each operation up to 100 times before failing.  

To use the sbsync library first create a SBTreeRoot object:

    tree_root = SBTreeRoot('test_folder', 'spaulinski@usgs.gov',
                           sb_root_folder_id='5fbe75fad34e4b9faad7e8a1')

This will prompt the user for a ScienceBase password.  Once entered you have access to the folder structure on ScienceBase through your SBTreeRoot object and its children/grandchildren/etc SBTreeNode objects. For example if the root folder contains a subfolder named "data", it can be accessed by:

    folder_under_tree_root = tree_root['data']
	
All subfolders of a SBTreeNode object can be accessed by its "folder_child_items" dictionary:

    for name, folder_obj in tree_root.folder_child_items.items():
	    print(folder_obj.sb_title)
		
Files are stored in SBFile objects and can be access in the same way as folders, either by specifying the file with brackets:

    data_file = tree_root['data']['data_file.csv']
   
Or by using the SBTreeNode "files" dictionary to access the SBFile object:

    for name, file_obj in folder_under_tree_root.files.items():
        print(file_obj.sb_name)
	   
Files and folders can be downloaded from and uploaded to ScienceBase. The contents of any SBTreeNode object can be populated by calling its "mirror_sciencebase_locally" method.  The copy_files parameter determines whether files along with the folder structure are copied locally.

    tree_root.mirror_sciencebase_locally(copy_files=True)

Uploading and downloading files can be done using the SBFile "upload_to_sciencebase" and "download_file_from_sciencebase" methods.

    data_file.upload_to_sciencebase()
	
Use the file_status property to determine if files are up to date, the local file is out of date, the ScienceBase file is out of date, or a merge is needed.

	fs = data_file.file_status
	
Whether a file is out of sync with ScienceBase is determined based on modification dates.  When a file is downloaded from ScienceBase the ScienceBase and local file modification dates are recorded in a file (_sync_log.csv).  The current ScienceBase and local file modification dates are compared against the recorded modification dates to determine the status of the file.

## sbtreeview.py
sbtreeview.py is a simple graphical user interface that uses the sbsync.py and pyqt5 python libraries.  This user interface can be used to help synchronize a ScienceBase archive with files and folders on your local computer.