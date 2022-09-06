import sciencebasepy
import time
from datetime import datetime
import sys
import os
import inspect
import csv
import hashlib
import ntpath
import enum
from collections import OrderedDict
from zipfile import ZipFile
from getpass import getpass


class FileStatus(enum.Enum):
    files_match = 1
    local_out_of_date = 2
    sciencebase_out_of_date = 3
    merge_required = 4
    local_file_missing = 5
    sciencebase_file_missing = 6
    both_files_missing = 7
    out_of_date_merge_status_unknown = 8


class DetailedLog:
    # ErrorLog is used to write errors out to a file to help with debugging
    # this python application
    _instance = None

    def __init__(self):
        raise RuntimeError("Call instance() instead")

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance._log_dict = None
            cls._instance._fd_log_file = open("detailed_log_sbsync.txt", "w")

        return cls._instance

    def write_log(self, message):
        self._fd_log_file.write(message)
        self._fd_log_file.write("\n\n")
        self._fd_log_file.flush()

    def __del__(self):
        self._fd_log_file.close()


class ErrorLog:
    # ErrorLog is used to write errors out to a file to help with debugging
    # this python application
    _instance = None

    def __init__(self):
        raise RuntimeError("Call instance() instead")

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance._log_dict = None
            cls._instance._fd_log_file = open("debug_log_sbsync.txt", "w")

        return cls._instance

    def write_log(
        self, err_class, err_method, stack, err_type, value, message=None
    ):
        self._fd_log_file.write(
            "---------------------------------------------"
        )
        self._fd_log_file.write(
            "---------------------------------------------"
        )
        self._fd_log_file.write(
            '\nAn error was caught in class "{}" method '
            '"{}"\nSee Debugging information below:'
            "\n".format(err_class, err_method)
        )
        self._fd_log_file.write("\nEXCEPTION TYPE: ")
        self._fd_log_file.write(err_type)
        self._fd_log_file.write("\nADDITIONAL INFORMATION: ")
        self._fd_log_file.write(value)
        self._fd_log_file.write("\n")
        if message is not None:
            self._fd_log_file.write(message)
            self._fd_log_file.write("\n")
        self._fd_log_file.write("\nCALL STACK\n---------------------\n")
        for entry in reversed(stack):
            self._fd_log_file.write(
                "{} (line {}): " "{}\n".format(entry[3], entry[2], entry[1])
            )

        self._fd_log_file.write("\n")
        self._fd_log_file.write(
            "---------------------------------------------"
        )
        self._fd_log_file.write(
            "---------------------------------------------"
        )
        self._fd_log_file.write("\n\n")
        self._fd_log_file.flush()

    def write_warning(self, message):
        self._fd_log_file.write(message)
        self._fd_log_file.write("\n\n")
        self._fd_log_file.flush()

    def __del__(self):
        self._fd_log_file.close()


class SyncLog:
    # SyncLog is used to produce a log of original file modification dates
    # and SHA1 hashes in order to detect merge conflicts.  The log contains a
    # line for each downloaded file:
    #       file_path, original_modification_date, SHA1 hash
    # SyncLog is a singleton class, only one instance of it can exist
    class LogFileInfo:
        def __init__(
            self,
            local_file_path,
            local_modification_date,
            sb_modification_date,
            checksum,
            zip_file_path,
        ):
            self.local_file_path = local_file_path
            self.local_modification_date = local_modification_date
            self.sb_modification_date = sb_modification_date
            self.checksum = checksum
            self.zip_file_path = zip_file_path

    _instance = None

    def __init__(self):
        raise RuntimeError("Call instance() instead")

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance._log_dict = {}
            cls._instance._zip_dict = {}
            cls._instance._log_file_path = None
        return cls._instance

    def set_log_path(self, local_project_path, log_name):
        self._log_file_path = os.path.join(local_project_path, log_name)
        self._load_log()

    def _load_log(self):
        try:
            self._log_dict = {}
            self._zip_dict = {}
            if os.path.isfile(self._log_file_path):
                with open(self._log_file_path, "r") as fd_read:
                    csv_rd = csv.reader(fd_read, delimiter=",", quotechar='"')
                    for line_lst in csv_rd:
                        local_mod_time = datetime.fromisoformat(line_lst[1])
                        sb_mod_time = datetime.fromisoformat(line_lst[2])
                        if len(line_lst) > 4:
                            zip_file_path = line_lst[4]
                        else:
                            zip_file_path = None
                        new_file_info = SyncLog.LogFileInfo(
                            line_lst[0],
                            local_mod_time,
                            sb_mod_time,
                            line_lst[3],
                            zip_file_path,
                        )
                        self._log_dict[line_lst[0]] = new_file_info
                        if zip_file_path in self._zip_dict:
                            self._zip_dict[zip_file_path].append(new_file_info)
                        else:
                            self._zip_dict[zip_file_path] = [new_file_info]

        except Exception as ex:
            type_, value_, traceback_ = sys.exc_info()
            stack = inspect.stack()
            ErrorLog.instance().write_log(
                "SyncLog", "_load_log", stack, str(type_), str(value_)
            )
            self._log_dict = {}
            self._zip_dict = {}

    def clear_log(self):
        if os.path.isfile(self._log_file_path):
            os.remove(self._log_file_path)
        self._log_dict = {}
        self._zip_dict = {}

    def update_log(
        self,
        file_path,
        local_modification_date,
        sb_modification_date,
        checksum,
        zip_file_path=None,
    ):
        new_info = SyncLog.LogFileInfo(
            file_path,
            local_modification_date,
            sb_modification_date,
            checksum,
            zip_file_path,
        )
        self._log_dict[file_path] = new_info
        if zip_file_path is not None:
            if zip_file_path in self._zip_dict:
                self._zip_dict[zip_file_path].append(new_info)
            else:
                self._zip_dict[zip_file_path] = [new_info]
        # update log file - for large folder structures this might be
        # costly to do every time
        self.write_log()

    def write_log(self):
        try:
            with open(self._log_file_path, "w") as fd_write:
                for file_path, file_info in self._log_dict.items():
                    fd_write.write(
                        "{},{},{},{}".format(
                            file_path,
                            file_info.local_modification_date,
                            file_info.sb_modification_date,
                            file_info.checksum,
                        )
                    )
                    if file_info.zip_file_path is None:
                        fd_write.write("\n")
                    else:
                        fd_write.write(",{}\n".format(file_info.zip_file_path))
        except Exception as ex:
            type_, value_, traceback_ = sys.exc_info()
            stack = inspect.stack()
            print("WARNING: Unable to write sync log.")
            ErrorLog.instance().write_log(
                "SyncLog", "_load_log", stack, str(type_), str(value_)
            )

    def get_zip_file_info(self, file_path):
        if file_path in self._zip_dict:
            return self._zip_dict[file_path]
        return None

    def get_file_info(self, file_path):
        if file_path in self._log_dict:
            return self._log_dict[file_path]
        return None


class SBAccess:
    def __init__(self, username=None, password=None):
        self.username = username
        self.password = password
        self.sb = sciencebasepy.SbSession()
        self.max_retries = 20
        self.authenticated = False
        self.log_all_messages = False
        self.too_many_retries = False
        if self.username:
            self._authenticate()

    def _write_detailed_log(self, method, text, params=None):
        if self.log_all_messages:
            if params is not None:
                if isinstance(params, list):
                    params_txt = ",".join(params)
                else:
                    params_txt = params
                message = "{} ({}): {}".format(method, text, params_txt)
            else:
                message = "{}: {}".format(method, text)
            DetailedLog.instance().write_log(message)

    def _remove_empty_json(self, jsnf):
        # get rid of any empty items
        remove_items = []
        for key, value in jsnf.items():
            if value is None:
                remove_items.append(key)
            if isinstance(value, dict):
                # recursively update dictionaries within this dictionary
                jsnf[key] = self._remove_empty_json(value)
            elif isinstance(value, list):
                # look for dictionaries in the list
                new_list = []
                for element in value:
                    if isinstance(element, dict):
                        # recursively update dictionary
                        new_list.append(self._remove_empty_json(element))
                    else:
                        new_list.append(element)
                jsnf[key] = new_list
        for item in remove_items:
            jsnf.pop(item)
        return jsnf

    def _handle_error(self, err, sb_id=None, std_err=True):
        if isinstance(err, ConnectionError):
            w_str = "WARNING: Connection aborted, reconnecting..."
            print(w_str)
            ErrorLog.instance().write_warning(w_str)
            self._authenticate()
        else:
            if sb_id is not None:
                id_text = " while getting item identified by {}".format(sb_id)
            else:
                id_text = ""
            if std_err:
                err_caught = ""
            else:
                err_caught = "Error not specifically trapped.  "
            if hasattr(err, "strerror"):
                error_name = err.strerror
            else:
                error_name = str(err)
            if hasattr(err, "errno"):
                error_num = err.errno
            else:
                error_num = "(No error number)"
            w_str = (
                'WARNING: "{}" occurred{}, '
                "error number {}.  {}Retrying.."
                ".".format(error_name, id_text, error_num, err_caught)
            )
            print(w_str)
            ErrorLog.instance().write_warning(w_str)

    def _handle_too_many_retries(self, op_name, *args):
        msg = "WARNING: Too many retry attempts. Unable to " "{}.".format(
            op_name
        )
        print(msg)
        msg = "{}  Parameters:".format(msg)
        if args:
            for value in args:
                msg = "{}\n{}".format(msg, value)
        ErrorLog.instance().write_warning(msg)
        self.too_many_retries = True

    def get_item(self, sb_id):
        success = False
        retry_attempt = 0
        item_json = None
        while not success and retry_attempt < self.max_retries:
            try:
                item_json = self.sb.get_item(sb_id)
                self._write_detailed_log("get_item", item_json, sb_id)
                success = True
            except OSError as err:
                self._write_detailed_log(
                    "get_item", "OSError Exception Raised", sb_id
                )
                self._handle_error(err, sb_id)
                retry_attempt += 1
            except Exception as err:
                self._write_detailed_log("get_item", "Exception Raised", sb_id)
                self._handle_error(err, sb_id, True)
                retry_attempt += 1
        if retry_attempt >= self.max_retries:
            self._handle_too_many_retries("get item")
        return item_json

    def get_child_ids(self, sb_id):
        success = False
        retry_attempt = 0
        child_ids = None
        while not success and retry_attempt < self.max_retries:
            try:
                child_ids = self.sb.get_child_ids(sb_id)
                self._write_detailed_log("get_child_ids", child_ids, sb_id)
                success = True
            except OSError as err:
                self._write_detailed_log(
                    "get_child_ids", "OSError Exception Raised", sb_id
                )
                self._handle_error(err, sb_id)
                retry_attempt += 1
            except Exception as err:
                self._write_detailed_log(
                    "get_child_ids", "Exception Raised", sb_id
                )
                self._handle_error(err, sb_id, False)
                retry_attempt += 1
        if retry_attempt >= self.max_retries:
            self._handle_too_many_retries("get child ids", sb_id)
        return child_ids

    def get_item_files(self, sb_json, destination):
        sb_json = self._remove_empty_json(sb_json)
        success = False
        retry_attempt = 0
        while not success and retry_attempt < self.max_retries:
            try:
                response = self.sb.get_item_files(sb_json, destination)
                self._write_detailed_log(
                    "get_item_files", response, [sb_json, destination]
                )
                success = True
            except OSError as err:
                self._write_detailed_log(
                    "get_item_files",
                    "OSError Exception Raised",
                    [sb_json, destination],
                )
                self._handle_error(err)
                retry_attempt += 1
            except Exception as err:
                self._write_detailed_log(
                    "get_item_files",
                    "Exception Raised",
                    [sb_json, destination],
                )
                self._handle_error(err, std_err=False)
                retry_attempt += 1
        if retry_attempt >= self.max_retries:
            self._handle_too_many_retries(
                "get item files", sb_json, destination
            )
        return success

    def download_file(self, sb_url, file_name, local_folder_path):
        success = False
        retry_attempt = 0
        while not success and retry_attempt < self.max_retries:
            try:
                complete_name = self.sb.download_file(
                    sb_url, file_name, local_folder_path
                )
                self._write_detailed_log(
                    "download_file",
                    complete_name,
                    [sb_url, file_name, local_folder_path],
                )
                success = True
            except OSError as err:
                self._write_detailed_log(
                    "download_file",
                    "OSError Exception Raised",
                    [sb_url, file_name, local_folder_path],
                )
                self._handle_error(err, sb_url)
                retry_attempt += 1
            except Exception as err:
                self._write_detailed_log(
                    "download_file",
                    "Exception Raised",
                    [sb_url, file_name, local_folder_path],
                )
                self._handle_error(err, sb_url, std_err=False)
                retry_attempt += 1
        if retry_attempt >= self.max_retries:
            self._handle_too_many_retries(
                "download file", sb_url, file_name, local_folder_path
            )
        return success

    def upload_files_to_item(
        self, json_item, local_file_paths, scrape_file=True
    ):
        json_item = self._remove_empty_json(json_item)
        success = False
        retry_attempt = 0
        while not success and retry_attempt < self.max_retries:
            try:
                rsp = self.sb.upload_files_and_update_item(
                    json_item, local_file_paths, scrape_file=scrape_file
                )
                self._write_detailed_log(
                    "upload_files_to_item",
                    rsp,
                    [json_item, local_file_paths, scrape_file],
                )
                success = True
            except OSError as err:
                self._write_detailed_log(
                    "upload_files_to_item",
                    "OSError Exception Raised",
                    [json_item, local_file_paths, scrape_file],
                )
                self._handle_error(err)
                retry_attempt += 1
            except Exception as err:
                self._write_detailed_log(
                    "upload_files_to_item",
                    "Exception Raised",
                    [json_item, local_file_paths, scrape_file],
                )
                self._handle_error(err, std_err=False)
                retry_attempt += 1
        if retry_attempt >= self.max_retries:
            self._handle_too_many_retries(
                "upload files to item", json_item, local_file_paths
            )
        return success

    def replace_file(self, local_file_path, json_item):
        json_item = self._remove_empty_json(json_item)
        success = False
        retry_attempt = 0
        while not success and retry_attempt < self.max_retries:
            try:
                value = self.sb.replace_file(local_file_path, json_item)
                self._write_detailed_log(
                    "replace_file", value, [local_file_path, json_item]
                )
                success = True
            except OSError as err:
                self._write_detailed_log(
                    "replace_file",
                    "OSError Exception Raised",
                    [local_file_path, json_item],
                )
                self._handle_error(err, local_file_path)
                retry_attempt += 1
            except Exception as err:
                self._write_detailed_log(
                    "replace_file",
                    "Exception Raised",
                    [local_file_path, json_item],
                )
                self._handle_error(err, local_file_path, std_err=False)
                retry_attempt += 1
        if retry_attempt >= self.max_retries:
            self._handle_too_many_retries(
                "replace file", local_file_path, json_item
            )
        return success

    def create_item(self, json_item):
        json_item = self._remove_empty_json(json_item)
        success = False
        retry_attempt = 0
        result = None
        while not success and retry_attempt < self.max_retries:
            try:
                result = self.sb.create_item(json_item)
                self._write_detailed_log("create_item", result, json_item)
                success = True
            except OSError as err:
                self._write_detailed_log(
                    "create_item", "OSError Exception Raised", json_item
                )
                self._handle_error(err)
                retry_attempt += 1
            except Exception as err:
                self._write_detailed_log(
                    "create_item", "Exception Raised", json_item
                )
                self._write_detailed_log("create_item", result, json_item)
                self._handle_error(err, std_err=False)
                retry_attempt += 1
        if retry_attempt >= self.max_retries:
            self._handle_too_many_retries("create item", json_item)
        return result

    def update_item(self, json_item):
        json_item = self._remove_empty_json(json_item)
        success = False
        retry_attempt = 0
        while not success and retry_attempt < self.max_retries:
            try:
                result = self.sb.update_item(json_item)
                self._write_detailed_log("update_item", result, json_item)
                success = True
            except OSError as err:
                self._write_detailed_log(
                    "update_item", "OSError Exception Raised", json_item
                )
                self._handle_error(err)
                retry_attempt += 1
            except Exception as err:
                self._write_detailed_log(
                    "update_item", "Exception Raised", json_item
                )
                self._write_detailed_log("update_item", result, json_item)
                self._handle_error(err, std_err=False)
                retry_attempt += 1
        if retry_attempt >= self.max_retries:
            self._handle_too_many_retries("update item", json_item)
        return success

    def _authenticate(self):
        if self.password is None:
            # getting password here so that it can be saved for later
            # authentication attempts
            self.password = getpass()
        try:
            self.sb.login(self.username, self.password)
            self.authenticated = True
        except Exception as ex:
            self.authenticated = False
            ErrorLog.instance().write_warning(
                "Failed to authenticate for "
                "username {}.  Exception: "
                "{}".format(self.username, ex)
            )

        # Need to wait a bit after the login or errors can occur
        time.sleep(5)


class SBPermissions:
    def __init__(
        self,
        json_txt=None,
        read_acl=None,
        read_inherited=None,
        read_inherits_from_id=None,
        write_acl=None,
        write_inherited=None,
        write_inherits_from_id=None,
    ):
        if json_txt is not None:
            if "read" in json_txt:
                json_read = json_txt["read"]
                if "acl" in json_read:
                    self.read_acl = json_read["acl"]
                if "inherited" in json_read:
                    self.read_inherited = json_read["inherited"]
                if "inheritsFromId" in json_read:
                    self.read_inherits_from_id = json_read["inheritsFromId"]
            if "write" in json_txt:
                json_write = json_txt["write"]
                if "acl" in json_write:
                    self.write_acl = json_write["acl"]
                if "inherited" in json_write:
                    self.write_inherited = json_write["inherited"]
                if "inheritsFromId" in json_write:
                    self.write_inherits_from_id = json_write["inheritsFromId"]
        else:
            self.read_acl = read_acl
            self.read_inherited = read_inherited
            self.read_inherits_from_id = read_inherits_from_id
            self.write_acl = write_acl
            self.write_inherited = write_inherited
            self.write_inherits_from_id = write_inherits_from_id

    @property
    def json_permissions(self):
        read_dict = {
            "acl": self.read_acl,
            "inherited": self.read_inherited,
            "inheritsFromId": self.read_inherits_from_id,
        }
        write_dict = {
            "acl": self.write_acl,
            "inherited": self.write_inherited,
            "inheritsFromId": self.write_inherits_from_id,
        }
        return {"read": read_dict, "write": write_dict}


class SBFile:
    """
    Class that contains ScienceBase and local file information.

    Parameters
    ----------
        sb_access : SBAccess
            access to the ScienceBase database
        parent_item : SBTreeNode
            path to the folder containing this file
        json_info : dict (json)
            part of json dictionary provided by ScienceBase containing file
            information

    Attributes
    ----------
        sb_cuid : str
            file's ScienceBase unique id.  ScienceBase assigns its own
            universally unique identifier to every item and uses it
            consistently throughout the architecture for all references. The
            UUID may be expressed as an HTTP URI (universal resource
            identifier) in some circumstances, but the basic ID is listed as a
            UUID string in the core model.
        sb_key : str
            File's ScienceBase key used as an identifier for the item.
        sb_bucket : ?
            ?
        sb_published : ?
            ?
        sb_node : ?
        sb_name : str
            The name of the file in ScienceBase.
        sb_title : str
            File's ScienceBase title.  Title is one of the most important
            ScienceBase attributes because it is used everywhere to give users
            a quick description of an item. ScienceBase has many different
            types of items coming from many sources, and titles need to be
            written so that they can be generally understood outside a
            particular context.

            Titles for some types of items such as citations harvested from a
            publication source are simply the full title of the item. Some
            titles, such as a title for a particular rock core in a large
            collection, may be constructed from various information about the
            item to build a readable and meaningful title.

            Titles are used as the item page's title so titles can show up in
            search results like Google.
        sb_content_type : str
            The ScienceBase type of file. Eg. "appliation/json".
        sb_content_encoding : str
            The ScienceBase encoding of the file.
        sb_path_on_disk : str
            The path on ScienceBase disk of the file. This is a string which
            can be used to download individual files.
        sb_processed : bool
            Whether the file has been processed by ScienceBase.
        sb_process_token : str
            The token used when processing this file.
        sb_image_width : int
            The width of the file if the file is an image (null otherwise).
        sb_image_height : int
            The height of the file if the file is an image (null otherwise).
        sb_size : int
            The size of the file on ScienceBase in bytes.
        sb_date_uploaded : datetime
            The date and time the file was uploaded to ScienceBase.
        sb_uploaded_by : str
            The email of the person who uploaded the file to ScienceBase.
        sb_original_metadata : bool
            Whether this is the original ScienceBase metadata of the file.
        sb_use_for_preview : bool
            Whether to use the file as a preview image. Only available for
            images.
        sb_s3_object : s3 Object?
            The s3 object with s3 download link and related information.
        sb_checksum : str
            SHA1 checksum recorded on ScienceBase
        sb_checksum_type : str
            Type of checksum used by ScienceBase (assumed to be SHA1)
        sb_url : str
            Url to ScienceBase file
        sb_download_uri : str
            Download uri for ScienceBase file
        sb_view_uri : str
            View uri for ScienceBase file
        local_file_path : str
            Path to file on your local computer
        local_file_exists : bool
            Whether file exists locally (has been downloaded) on your computer
        local_sha1_checksum : str
            Sha1 checksum of local file on your computer
        local_modify_time : datetime
            When the local file on your computer was last modified
        sb_file_exists : bool
            Whether file exists on ScienceBase
        checksum_matches : bool
            Whether local file checksum matches ScienceBase file checksum
        modify_dates_match : bool
            Whether local file modify date matches ScienceBase file modify date

    Methods
    -------

    See Also
    --------

    Notes
    -----

    Examples
    --------


    """

    def __init__(self, sb_access, parent_item, json_info):
        self._sb_access = sb_access
        self._parent_item = parent_item
        self.zip_files = None
        self.local_folder_path = parent_item.local_folder_path
        self.sb_cuid = None
        self.sb_key = None
        self.sb_bucket = None
        self.sb_published = None
        self.sb_node = None
        self.sb_name = None
        self.sb_title = None
        self.sb_content_type = None
        self.sb_content_encoding = None
        self.sb_path_on_disk = None
        self.sb_processed = None
        self.sb_process_token = None
        self.sb_image_width = None
        self.sb_image_height = None
        self.sb_size = None
        self.sb_date_uploaded = None
        self.sb_uploaded_by = None
        self.sb_original_metadata = None
        self.sb_use_for_preview = None
        self.sb_s3_object = None
        self.sb_checksum = None
        self.sb_checksum_type = None
        self.sb_url = None
        self.sb_download_uri = None
        self.sb_view_uri = None
        self.refresh(json_info)

    def refresh(self, json_info):
        try:
            if json_info is not None:
                if "cuid" in json_info:
                    self.sb_cuid = json_info["cuid"]
                if "key" in json_info:
                    self.sb_key = json_info["key"]
                if "bucket" in json_info:
                    self.sb_bucket = json_info["bucket"]
                if "published" in json_info:
                    self.sb_published = json_info["published"]
                if "node" in json_info:
                    self.sb_node = json_info["node"]
                if "name" in json_info:
                    self.sb_name = json_info["name"]
                if "title" in json_info:
                    self.sb_title = json_info["title"]
                if "contentType" in json_info:
                    self.sb_content_type = json_info["contentType"]
                if "contentEncoding" in json_info:
                    self.sb_content_encoding = json_info["contentEncoding"]
                if "pathOnDisk" in json_info:
                    self.sb_path_on_disk = json_info["pathOnDisk"]
                if "processed" in json_info:
                    self.sb_processed = json_info["processed"]
                if "processToken" in json_info:
                    self.sb_process_token = json_info["processToken"]
                if "imageWidth" in json_info:
                    self.sb_image_width = json_info["imageWidth"]
                if "imageHeight" in json_info:
                    self.sb_image_height = json_info["imageHeight"]
                if "size" in json_info:
                    self.sb_size = json_info["size"]
                if "dateUploaded" in json_info:
                    if json_info["dateUploaded"] is not None:
                        self.sb_date_uploaded = self._to_python_datetime(
                            json_info["dateUploaded"]
                        )
                    else:
                        self.sb_date_uploaded = None
                if "uploadedBy" in json_info:
                    self.sb_uploaded_by = json_info["uploadedBy"]
                if "originalMetadata" in json_info:
                    self.sb_original_metadata = json_info["originalMetadata"]
                if "useForPreview" in json_info:
                    self.sb_use_for_preview = json_info["useForPreview"]
                if "s3Object" in json_info:
                    self.sb_s3_object = json_info["s3Object"]
                if (
                    "checksum" in json_info
                    and json_info["checksum"] is not None
                ):
                    if "value" in json_info["checksum"]:
                        self.sb_checksum = json_info["checksum"]["value"]
                    if "type" in json_info["checksum"]:
                        self.sb_checksum_type = json_info["checksum"]["type"]
                if "url" in json_info:
                    self.sb_url = json_info["url"]
                if "downloadUri" in json_info:
                    self.sb_download_uri = json_info["downloadUri"]
                if "viewUri" in json_info:
                    self.sb_view_uri = json_info["viewUri"]
            self._load_zip_dict()
        except Exception as ex:
            # write uncaught exception to log and then raise it again
            type_, value_, traceback_ = sys.exc_info()
            full_stack = inspect.stack()
            message = None
            if self.sb_name is not None:
                message = "Error occurred in SBFile with name: " "{}".format(
                    self.sb_name
                )
            ErrorLog.instance().write_log(
                "SBFile",
                "refresh",
                full_stack,
                str(type_),
                str(value_),
                message,
            )
            raise ex

    @staticmethod
    def _local_json(file_name):
        return {
            "cuid": None,
            "key": None,
            "bucket": None,
            "published": None,
            "node": None,
            "name": file_name,
            "title": file_name,
            "contentType": None,
            "contentEncoding": None,
            "pathOnDisk": None,
            "processed": None,
            "processToken": None,
            "imageWidth": None,
            "imageHeight": None,
            "size": None,
            "dateUploaded": None,
            "uploadedBy": None,
            "originalMetadata": None,
            "useForPreview": None,
            "s3Object": None,
            "checksum": {"value": None, "type": None},
            "url": None,
            "downloadUri": None,
            "viewUri": None,
        }

    @classmethod
    def local_file(cls, sb_access, parent_item, local_file_path):
        """Create a SBFile object from a local file.

        Parameters
        ----------
        sb_access : SBAccess
            access to ScienceBase
        parent_item : SBTreeNode
            parent folder object
        local_file_path : string
            path to local file

        Returns
        -------
        sbfile : SBFile object

        Examples
        --------
        >>> s = SBFile.local_file(os.path.join('folder', 'file.txt'))

        """
        folder_path, file_name = ntpath.split(local_file_path)
        return cls(sb_access, parent_item, SBFile._local_json(file_name))

    def json_file(self, basics_only=False):
        if basics_only:
            return {
                "name": self.sb_name,
                "title": self.sb_title,
                "contentType": self.sb_content_type,
                "pathOnDisk": self.sb_path_on_disk,
            }

        jsnf = {
            "cuid": self.sb_cuid,
            "key": self.sb_key,
            "bucket": self.sb_bucket,
            "published": self.sb_published,
            "node": self.sb_node,
            "name": self.sb_name,
            "title": self.sb_title,
            "contentType": self.sb_content_type,
            "contentEncoding": self.sb_content_encoding,
            "pathOnDisk": self.sb_path_on_disk,
            "processed": self.sb_processed,
            "processToken": self.sb_process_token,
            "imageWidth": self.sb_image_width,
            "imageHeight": self.sb_image_height,
            "size": self.sb_size,
            "dateUploaded": self._to_sb_datetime(self.sb_date_uploaded),
            "uploadedBy": self.sb_uploaded_by,
            "originalMetadata": self.sb_original_metadata,
            "useForPreview": self.sb_use_for_preview,
            "s3Object": self.sb_s3_object,
            "url": self.sb_url,
            "downloadUri": self.sb_download_uri,
            "viewUri": self.sb_view_uri,
        }
        if self.sb_checksum is not None and self.sb_checksum_type is not None:
            jsnf["checksum"] = (
                {"value": self.sb_checksum, "type": self.sb_checksum_type},
            )
        return jsnf

    @property
    def local_file_path(self):
        return os.path.join(self.local_folder_path, self.sb_name)

    @property
    def local_file_exists(self):
        return os.path.isfile(self.local_file_path)

    @property
    def is_zip_file(self):
        name_parts = self.sb_name.split(".")
        if len(name_parts) > 1 and name_parts[-1].lower() == "zip":
            return True
        return False

    def _hash_binary_file(self, fd):
        try:
            BLOCKSIZE = 65536
            file_hash = hashlib.md5()
            buf = fd.read(BLOCKSIZE)
            while len(buf) > 0:
                file_hash.update(buf)
                buf = fd.read(BLOCKSIZE)
            return file_hash.hexdigest()
        except Exception as ex:
            # write uncaught exception to log and then raise it again
            type_, value_, traceback_ = sys.exc_info()
            full_stack = inspect.stack()
            message = None
            if self.sb_name is not None:
                message = "Error occurred in SBFile with name: " "{}".format(
                    self.sb_name
                )
            ErrorLog.instance().write_log(
                "SBFile",
                "_hash_binary_file",
                full_stack,
                str(type_),
                str(value_),
                message,
            )
            raise ex

    @property
    def local_checksum(self):
        try:
            fd = open(self.local_file_path, "rb")
            hash_text = self._hash_binary_file(fd)
            fd.close()
            return hash_text
        except Exception as ex:
            # write uncaught exception to log and then raise it again
            type_, value_, traceback_ = sys.exc_info()
            full_stack = inspect.stack()
            message = None
            if self.sb_name is not None:
                message = "Error occurred in SBFile with name: " "{}".format(
                    self.sb_name
                )
            ErrorLog.instance().write_log(
                "SBFile",
                "local_checksum",
                full_stack,
                str(type_),
                str(value_),
                message,
            )
            raise ex

    @property
    def local_modify_time(self):
        return self._get_modify_time(self.local_file_path)

    @staticmethod
    def _get_modify_time(local_file_path):
        try:
            if os.path.isfile(local_file_path):
                return datetime.fromtimestamp(
                    os.path.getmtime(local_file_path)
                )
            return None
        except Exception as ex:
            # write uncaught exception to log and then raise it again
            type_, value_, traceback_ = sys.exc_info()
            full_stack = inspect.stack()
            message = None
            if local_file_path is not None:
                message = "Error occurred in SBFile with path: " "{}".format(
                    local_file_path
                )
            ErrorLog.instance().write_log(
                "SBFile",
                "_get_modify_time",
                full_stack,
                str(type_),
                str(value_),
                message,
            )
            raise ex

    @property
    def sb_file_exists(self):
        return self.sb_path_on_disk is not None

    @property
    def checksum_matches(self):
        return self.local_checksum == self.sb_checksum

    @property
    def modify_dates_match(self):
        return self.local_modify_time == self.sb_date_uploaded

    @property
    def file_status(self):
        try:
            if not self.sb_file_exists:
                if not self.local_file_exists:
                    return FileStatus.both_files_missing
                else:
                    return FileStatus.sciencebase_file_missing
            elif not self.local_file_exists:
                return FileStatus.local_file_missing
            else:
                if self.is_zip_file:
                    if self.zip_files is None:
                        self._load_zip_dict()
                    last_status = None
                    for file in self.zip_files.keys():
                        status = self._sync_log_file_status(
                            file, self._get_modify_time(file)
                        )
                        if last_status is None:
                            last_status = status
                        elif (
                            status
                            == FileStatus.out_of_date_merge_status_unknown
                            or status == FileStatus.merge_required
                            or status == FileStatus.local_out_of_date
                        ):
                            last_status == status
                        elif status == FileStatus.sciencebase_out_of_date:
                            if last_status == FileStatus.files_match:
                                last_status = status
                            elif last_status == FileStatus.local_out_of_date:
                                last_status = FileStatus.merge_required
                    return last_status
                else:
                    return self._sync_log_file_status(
                        self.local_file_path, self.local_modify_time
                    )
        except Exception as ex:
            # write uncaught exception to log and then raise it again
            type_, value_, traceback_ = sys.exc_info()
            full_stack = inspect.stack()
            message = None
            if self.sb_name is not None:
                message = "Error occurred in SBFile with name: " "{}".format(
                    self.sb_name
                )
            ErrorLog.instance().write_log(
                "SBFile",
                "file_status",
                full_stack,
                str(type_),
                str(value_),
                message,
            )
            raise ex

    def _load_zip_dict(self):
        self.zip_files = {}
        zip_file_info = SyncLog.instance().get_zip_file_info(
            self.local_file_path
        )
        if zip_file_info is not None:
            for item in zip_file_info:
                self.zip_files[item.local_file_path] = True

    def _sync_log_file_status(self, file_path, modify_time):
        orig_info = SyncLog.instance().get_file_info(file_path)
        if orig_info is None:
            return FileStatus.out_of_date_merge_status_unknown
        local_file_modified = orig_info.local_modification_date < modify_time

        server_file_modified = (
            orig_info.sb_modification_date < self.sb_date_uploaded
        )
        if local_file_modified:
            if server_file_modified:
                return FileStatus.merge_required
            else:
                return FileStatus.sciencebase_out_of_date
        elif server_file_modified:
            return FileStatus.local_out_of_date
        else:
            return FileStatus.files_match

    def upload_to_sciencebase(self, scrape_file=True, convert_to_zip=False):
        try:
            local_file_path = [self.local_file_path]
            zip_file_path = None
            if self.sb_file_exists:
                assert not convert_to_zip
                if self.is_zip_file:
                    # remove old zip file
                    if os.path.exists(self.local_file_path):
                        os.remove(self.local_file_path)
                    # create updated zip file
                    local_file_path = []
                    with ZipFile(self.local_file_path, "w") as zip_archive:
                        for file in self.zip_files.keys():
                            zip_archive.write(file, os.path.basename(file))
                            local_file_path.append(file)
                    zip_file_path = self.local_file_path
                # replace item
                success = self._sb_access.replace_file(
                    self.local_file_path,
                    self._parent_item.json_folder(True, True),
                )
            else:
                if convert_to_zip:
                    self.zip_files = {self.local_file_path: True}
                    # update path to zip file
                    sb_old_name = self.sb_name
                    file_base = os.path.splitext(self.sb_name)[0]
                    self.sb_name = "{}.zip".format(file_base)
                    # create zip file
                    with ZipFile(self.local_file_path, "w") as zip_archive:
                        for file in self.zip_files.keys():
                            zip_archive.write(file, os.path.basename(file))
                    # update parent with zip file name
                    if sb_old_name in self._parent_item.files:
                        del self._parent_item.files[sb_old_name]
                    self._parent_item.files[self.sb_name] = self
                    zip_file_path = self.local_file_path
                # make sure containing folder exists on ScienceBase
                if not self._parent_item.sb_folder_exists:
                    self._parent_item.upload_to_sciencebase()
                # create item
                success = self._sb_access.upload_files_to_item(
                    self._parent_item.json_folder(),
                    [self.local_file_path],
                    scrape_file=scrape_file,
                )
            if success:
                # resync with server
                self._parent_item.refresh()
                for path in local_file_path:
                    SyncLog.instance().update_log(
                        path,
                        self.local_modify_time,
                        self.sb_date_uploaded,
                        self.local_checksum,
                        zip_file_path,
                    )

            return success
        except Exception as ex:
            # write uncaught exception to log and then raise it again
            type_, value_, traceback_ = sys.exc_info()
            full_stack = inspect.stack()
            message = None
            if self.sb_name is not None:
                message = "Error occurred in SBFile with name: " "{}".format(
                    self.sb_name
                )
            ErrorLog.instance().write_log(
                "SBFile",
                "upload_to_sciencebase",
                full_stack,
                str(type_),
                str(value_),
                message,
            )
            raise ex

    def extract_zip(self):
        # unzip file
        with ZipFile(self.local_file_path, "r") as zip_file:
            zip_file.extractall(self.local_folder_path)
            zipped_files = zip_file.namelist()
        # store names of all unziped files and update file log
        self.zip_files = {}
        sync_log = SyncLog.instance()
        if len(zipped_files) > 1:
            # currently not supporting multiple files
            message = (
                "Zip files containing multiple files is currently "
                "not supported.  Only zip files with a single "
                "file supported.  File {} contains multiple files"
                ".".format(self.sb_name)
            )
            type_, value_, traceback_ = sys.exc_info()
            full_stack = inspect.stack()
            ErrorLog.instance().write_log(
                "SBFile",
                "upload_to_sciencebase",
                full_stack,
                str(type_),
                str(value_),
                message,
            )
            raise Exception(message)
        for zipped_file in zipped_files:
            zf_full_path = os.path.join(self.local_folder_path, zipped_file)
            self.zip_files[zf_full_path] = True
            sync_log.update_log(
                zf_full_path,
                self._get_modify_time(zf_full_path),
                self.sb_date_uploaded,
                self.local_checksum,
                self.local_file_path,
            )

    def download_file_from_sciencebase(self):
        try:
            success = self._sb_access.download_file(
                self.sb_url, self.sb_name, self.local_folder_path
            )
            if success:
                if self.is_zip_file:
                    self.extract_zip()
                else:
                    self.zip_files = None
                    # resync with server
                    SyncLog.instance().update_log(
                        self.local_file_path,
                        self.local_modify_time,
                        self.sb_date_uploaded,
                        self.local_checksum,
                    )
            self._parent_item.refresh()
            return success
        except Exception as ex:
            # write uncaught exception to log and then raise it again
            type_, value_, traceback_ = sys.exc_info()
            full_stack = inspect.stack()
            message = None
            if self.sb_name is not None:
                message = "Error occurred in SBFile with name: " "{}".format(
                    self.sb_name
                )
            ErrorLog.instance().write_log(
                "SBFile",
                "download_file_from_sciencebase",
                full_stack,
                str(type_),
                str(value_),
                message,
            )
            raise ex

    @staticmethod
    def _to_sb_datetime(date_time):
        if date_time is not None:
            return "{}-{}-{}T{}:{}:{}" "Z".format(
                date_time.year,
                date_time.month,
                date_time.day,
                date_time.hour,
                date_time.minute,
                date_time.second,
            )
        return None

    @staticmethod
    def _to_python_datetime(sb_date):
        return datetime(
            year=int(sb_date[0:4]),
            month=int(sb_date[5:7]),
            day=int(sb_date[8:10]),
            hour=int(sb_date[11:13]),
            minute=int(sb_date[14:16]),
            second=int(sb_date[17:19]),
        )


class SBTreeNode:
    """
    Class that contains ScienceBase and local file information.

    Parameters
    ----------
        parent_folder_path : str
            Path of parent folder
        sb_access : SBAccess
            Access to the ScienceBase database
        parent_item : SBTreeNode
            Path to the folder containing this file
        sb_id : str
            Folder's ScienceBase unique id
        title : str
            Folder's title (name)

    Attributes
    ----------
        sb_id : str
            folder's ScienceBase unique id.  ScienceBase assigns its own
            universally unique identifier to every item and uses it
            consistently throughout the architecture for all references. The
            UUID may be expressed as an HTTP URI (universal resource
            identifier) in some circumstances, but the basic ID is listed as a
            UUID string in the core model.
        sb_title : str
            Folder's ScienceBase title.  Title is one of the most important
            ScienceBase attributes because it is used everywhere to give users
            a quick description of an item. ScienceBase has many different
            types of items coming from many sources, and titles need to be
            written so that they can be generally understood outside a
            particular context.

            Titles for some types of items such as citations harvested from a
            publication source are simply the full title of the item. Some
            titles, such as a title for a particular rock core in a large
            collection, may be constructed from various information about the
            item to build a readable and meaningful title.

            Titles are used as the item page's title so titles can show up in
            search results like Google.
        sb_has_children : bool
            folder contains sub-folders
        sb_data_source : ?
            ?
        sb_last_updated_by : str
            The email of the person who uploaded the file to ScienceBase.
        sb_last_updated_date : datetime
            The date and time the file was uploaded to ScienceBase.
        sb_date_created : datetime
            The date and time the file was created in ScienceBase.
        sb_created_by : str
            The email of the person who created the file to ScienceBase.
        sb_permissions : SBPermissions
            Folder's permissions on ScienceBase
        files : dict
            Dictionary of files contained in this folder, key is file name,
            value is SBFile
        sb_is_child : bool
            Is folder a child folder
        sb_link : str
            ?
        sb_relatedItems : ?
            ?
        sb_provenance : ?
            ?
        sb_parentId : str
            ID of parent folder
        sb_systemTypes : ?
            ?
        sb_distributionLinks : ?
            ?
        sb_locked : ?
            ?
        sb_folder_exists : bool
            Folder exists on ScienceBase
        local_folder_path : str
            Local path to this folder

    Methods
    -------
    json_folder(include_files=False) : dict
        Folder's json information.  When include_files is True
        includes file information.
    refresh()
        Refresh folder information and information of any files
        contained in this folder
    copy_item_files_local()
        Copy files in this ScienceBase folder to the local drive
    mirror_sciencebase_locally(copyfiles=False)
        Recursively mirror folders and subfolders from sciencebase
        on the local drive.  When copyfiles is True all files in
        folders and subfolders are copied from ScienceBase to
        the local drive.
    upload_to_sciencebase()
        Create or update folder information on ScienceBase
    populate_local_folder_structure()
        Populate SBTreeNode structure with local files and folders
        not yet uploaded to ScienceBase

    See Also
    --------

    Notes
    -----

    Examples
    --------


    """

    MAX_FOLDER_NAME_LENGTH = 50

    def __init__(
        self, parent_folder_path, sb_access, parent_item, sb_id, title=None
    ):
        self.sb_access = sb_access
        self.parent_item = parent_item
        self.folder_child_items = OrderedDict()
        self.sb_title = title
        self.sb_id = sb_id
        self.sb_has_children = None
        self.local_has_children = None
        self.sb_data_source = None
        self.sb_last_updated_by = None
        self.sb_last_updated_date = None
        self.sb_date_created = None
        self.sb_created_by = None
        self.sb_permissions = None
        self.files = OrderedDict()
        self.sb_is_child = None
        self._json_txt = None

        self.sb_link = None
        self.sb_relatedItems = None
        self.sb_provenance = None
        if parent_item is not None:
            self.sb_parentId = parent_item.sb_id
        else:
            self.sb_parentId = None
        self.sb_systemTypes = None
        self.sb_distributionLinks = None
        self.sb_locked = None

        self.local_parent_folder_path = parent_folder_path
        if self.sb_id is not None:
            self.refresh()

    def __getitem__(self, k):
        assert isinstance(k, str)
        if k in self.folder_child_items:
            return self.folder_child_items[k]
        elif k in self.files:
            return self.files[k]
        elif k == "..":
            return self.parent_item
        return None

    def refresh(self, item_json=None):
        try:
            if item_json is None:
                item_json = self.sb_access.get_item(self.sb_id)
            if not isinstance(item_json, dict):
                print(
                    "WARNING: No JSON information returned for item "
                    "{}.".format(str(self.sb_id))
                )
                return
            if "id" in item_json:
                self.sb_id = item_json["id"]
            if "title" in item_json:
                self.sb_title = item_json["title"]
            if "hasChildren" in item_json:
                self.sb_has_children = item_json["hasChildren"]
            if "provenance" in item_json:
                prov = item_json["provenance"]
                if "dataSource" in prov:
                    self.sb_data_source = prov["dataSource"]
                if "lastUpdatedBy" in prov:
                    self.sb_last_updated_by = prov["lastUpdatedBy"]
                if "lastUpdated" in prov:
                    self.sb_last_updated_date = prov["lastUpdated"]
                if "dateCreated" in prov:
                    self.sb_date_created = prov["dateCreated"]
                if "createdBy" in prov:
                    self.sb_created_by = prov["createdBy"]
            if "permissions" in item_json:
                self.sb_permissions = SBPermissions(item_json["permissions"])
            if "link" in item_json:
                self.sb_link = item_json["link"]
            if "relatedItems" in item_json:
                self.sb_relatedItems = item_json["relatedItems"]
            if "parentId" in item_json:
                self.sb_parentId = item_json["parentId"]
            if "systemTypes" in item_json:
                self.sb_systemTypes = item_json["systemTypes"]
            else:
                self.sb_systemTypes = None
            if "distributionLinks" in item_json:
                self.sb_distributionLinks = item_json["distributionLinks"]
            if "locked" in item_json:
                self.sb_locked = item_json["locked"]
            self._json_txt = item_json

            if "files" in item_json:
                for file_json in item_json["files"]:
                    new_file = SBFile(self.sb_access, self, file_json)
                    if new_file.sb_name in self.files:
                        self.files[new_file.sb_name].refresh(file_json)
                    else:
                        self.files[new_file.sb_name] = new_file
            # sort
            self._sort_items()
        except Exception as ex:
            # write uncaught exception to log and then raise it again
            type_, value_, traceback_ = sys.exc_info()
            full_stack = inspect.stack()
            message = None
            if self.local_folder_path is not None:
                message = (
                    "Error occurred in SBTreeNode with path: "
                    "{}".format(self.local_folder_path)
                )
            ErrorLog.instance().write_log(
                "SBTreeNode",
                "refresh",
                full_stack,
                str(type_),
                str(value_),
                message,
            )
            raise ex

    @property
    def sb_folder_exists(self):
        return self.sb_id is not None

    def copy_item_files_local(self):
        try:
            if self.local_folder_path is None:
                raise Exception(
                    'Error: you must first call "mirror_local" to '
                    "mirror the folder structure locally"
                )
            if len(self.files) > 0:
                # download files from ScienceBase
                if not self.sb_access.get_item_files(
                    self._json_txt, self.local_folder_path
                ):
                    raise Exception(
                        "Error: failed to get files from "
                        "{}".format(self.sb_title)
                    )
                for file in self.files.values():
                    # sync local modification dates with ScienceBase
                    # record in sync log
                    if file.is_zip_file:
                        file.extract_zip()
                    else:
                        SyncLog.instance().update_log(
                            file.local_file_path,
                            file.local_modify_time,
                            file.sb_date_uploaded,
                            file.local_checksum,
                        )
        except Exception as ex:
            # write uncaught exception to log and then raise it again
            type_, value_, traceback_ = sys.exc_info()
            full_stack = inspect.stack()
            message = None
            if self.local_folder_path is not None:
                message = (
                    "Error occurred in SBTreeNode with path: "
                    "{}".format(self.local_folder_path)
                )
            ErrorLog.instance().write_log(
                "SBTreeNode",
                "copy_item_files_local",
                full_stack,
                str(type_),
                str(value_),
                message,
            )
            raise ex

    @property
    def json_provenace(self):
        jsprov = {}
        if self.sb_data_source is not None:
            jsprov["dataSource"] = self.sb_data_source
        if self.sb_last_updated_by is not None:
            jsprov["lastUpdatedBy"] = self.sb_last_updated_by
        if self.sb_last_updated_date is not None:
            jsprov["lastUpdated"] = self.sb_last_updated_date
        if self.sb_data_source is not None:
            jsprov["dateCreated"] = self.sb_data_source
        if self.sb_created_by is not None:
            jsprov["createdBy"] = self.sb_created_by
        return jsprov

    def json_folder(self, include_files=False, basics_only=False):
        try:
            if basics_only:
                jsnf = {"id": self.sb_id, "title": self.sb_title}
            else:
                if self.sb_permissions is None:
                    json_permissions = None
                else:
                    json_permissions = self.sb_permissions.json_permissions
                jsnf = {
                    "link": self.sb_link,
                    "relatedItems": self.sb_relatedItems,
                    "id": self.sb_id,
                    "title": self.sb_title,
                    "provenance": self.json_provenace,
                    "hasChildren": self.sb_has_children,
                    "parentId": self.sb_parentId,
                    "systemTypes": self.sb_systemTypes,
                    "permissions": json_permissions,
                    "distributionLinks": self.sb_distributionLinks,
                    "locked": self.sb_locked,
                }
            if include_files:
                files = []
                for file in self.files.values():
                    if file.sb_path_on_disk is not None:
                        files.append(file.json_file(basics_only))
                jsnf["files"] = files
            return jsnf
        except Exception as ex:
            # write uncaught exception to log and then raise it again
            type_, value_, traceback_ = sys.exc_info()
            full_stack = inspect.stack()
            message = None
            if self.local_folder_path is not None:
                message = (
                    "Error occurred in SBTreeNode with path: "
                    "{}".format(self.local_folder_path)
                )
            ErrorLog.instance().write_log(
                "SBTreeNode",
                "json_folder",
                full_stack,
                str(type_),
                str(value_),
                message,
            )
            raise ex

    def upload_to_sciencebase(self):
        try:
            if self.sb_folder_exists:
                result = self.sb_access.update_item(self.json_folder())
                self.refresh()
            else:
                # make sure parent folder exists on sciencebase
                if self.parent_item and not self.parent_item.sb_folder_exists:
                    self.parent_item.upload_to_sciencebase()
                    self.sb_parentId = self.parent_item.sb_id
                # upload folder to sciencebase
                result = self.sb_access.create_item(self.json_folder())
                self.refresh(result)
            return result
        except Exception as ex:
            # write uncaught exception to log and then raise it again
            type_, value_, traceback_ = sys.exc_info()
            full_stack = inspect.stack()
            message = None
            if self.local_folder_path is not None:
                message = (
                    "Error occurred in SBTreeNode with path: "
                    "{}".format(self.local_folder_path)
                )
            ErrorLog.instance().write_log(
                "SBTreeNode",
                "upload_to_sciencebase",
                full_stack,
                str(type_),
                str(value_),
                message,
            )
            raise ex

    def populate_local_folder_structure(self, recursive_populate=True):
        try:
            # build dictionary of files belonging to a zip archive
            zip_archive_files = {}
            for file in self.files.values():
                if file.is_zip_file and file.zip_files is not None:
                    zip_archive_files.update(file.zip_files)

            # get folder contents of next folder
            child_items = os.listdir(self.local_folder_path)
            for child_item in child_items:
                child_path = os.path.join(self.local_folder_path, child_item)
                if os.path.isfile(child_path):
                    if (
                        child_item not in self.files
                        and child_path not in zip_archive_files
                    ):
                        # add file to structure
                        self.files[child_item] = SBFile.local_file(
                            self.sb_access, self, child_path
                        )
                else:
                    if child_item not in self.folder_child_items:
                        self.folder_child_items[child_item] = SBTreeNode(
                            self.local_folder_path,
                            self.sb_access,
                            self,
                            None,
                            child_item,
                        )
                        self.local_has_children = True
            if recursive_populate:
                for child_item in self.folder_child_items.values():
                    child_item.populate_local_folder_structure()
            # sort
            self._sort_items()
        except Exception as ex:
            # write uncaught exception to log and then raise it again
            type_, value_, traceback_ = sys.exc_info()
            full_stack = inspect.stack()
            message = None
            if self.local_folder_path is not None:
                message = (
                    "Error occurred in SBTreeNode with path: "
                    "{}".format(self.local_folder_path)
                )
            ErrorLog.instance().write_log(
                "SBTreeNode",
                "populate_local_folder_structure",
                full_stack,
                str(type_),
                str(value_),
                message,
            )
            raise ex

    def _sort_items(self):
        self.folder_child_items = OrderedDict(
            sorted(self.folder_child_items.items(), key=lambda t: t[0])
        )
        self.files = OrderedDict(
            sorted(self.files.items(), key=lambda t: t[0])
        )

    def mirror_sciencebase_locally(self, copy_files=False):
        try:
            if not os.path.exists(self.local_folder_path):
                os.mkdir(self.local_folder_path)
            if copy_files:
                self.copy_item_files_local()
            for child_item in self.folder_child_items.values():
                child_item.mirror_sciencebase_locally(copy_files)
        except Exception as ex:
            # write uncaught exception to log and then raise it again
            type_, value_, traceback_ = sys.exc_info()
            full_stack = inspect.stack()
            message = None
            if self.local_folder_path is not None:
                message = (
                    "Error occurred in SBTreeNode with path: "
                    "{}".format(self.local_folder_path)
                )
            ErrorLog.instance().write_log(
                "SBTreeNode",
                "mirror_sciencebase_locally",
                full_stack,
                str(type_),
                str(value_),
                message,
            )
            raise ex

    @property
    def local_folder_path(self):
        if self.sb_title is None:
            return self.local_parent_folder_path
        else:
            if len(self.sb_title) > SBTreeNode.MAX_FOLDER_NAME_LENGTH:
                sb_title = "{}...{}".format(
                    self.sb_title[:37], self.sb_title[-10:]
                )
            else:
                sb_title = self.sb_title
            return os.path.join(self.local_parent_folder_path, sb_title)

    def create_item(self):
        return


class SBTreeRoot(SBTreeNode):
    """
    Class that contains ScienceBase and local file information.

    Parameters
    ----------
        folder_path : str
            Path of root folder
        username : str
            Username to log into ScienceBase
        password : str
            Password to log into ScienceBase.  If not specified
            user will be prompted.
        sb_root_folder_id : str
            Root folder's ScienceBase unique id

    Attributes
    ----------
        root_folder_id : str
            root folder's ScienceBase unique id.  ScienceBase assigns its own
            universally unique identifier to every item and uses it
            consistently throughout the architecture for all references. The
            UUID may be expressed as an HTTP URI (universal resource
            identifier) in some circumstances, but the basic ID is listed as a
            UUID string in the core model.

    Methods
    -------
    refresh()
        Refresh folder information and information of any files
        contained in this folder
    copy_item_files_local()
        Copy files in this ScienceBase folder to the local drive
    read_sb_folder_structure()
        Reads the folder structure from ScienceBase into SBTreeNode
        objects.
    mirror_sciencebase_locally(copyfiles=False,
                               replace_local_copy=False)
        Recursively mirror folders and subfolders from sciencebase
        on the local drive.  Files are mirrored to "folder_path",
        which is specified as one of the initialization parameters.
        When copyfiles is True all files in folders and subfolders
        are copied from ScienceBase to the local drive.  When
        replace_local_copy is specified the entire local folder
        structure specified by "folder_path" is deleted first.

    See Also
    --------

    Notes
    -----

    Examples
    --------


    """

    def __init__(
        self,
        folder_path,
        username=None,
        password=None,
        sb_root_folder_id="5ed6379982ce7e579c6494b6",
        log_name="_sync_log.csv",
    ):
        try:
            self.root_folder_id = sb_root_folder_id
            sb_access = SBAccess(username, password)
            self.authenticated = sb_access.authenticated
            if sb_access.authenticated:
                super(SBTreeRoot, self).__init__(
                    folder_path, sb_access, None, sb_root_folder_id
                )
                sync_log = SyncLog.instance()
                sync_log.set_log_path(folder_path, log_name)
                self.read_sb_folder_structure()
        except Exception as ex:
            # write uncaught exception to log and then raise it again
            type_, value_, traceback_ = sys.exc_info()
            full_stack = inspect.stack()
            ErrorLog.instance().write_log(
                "SBTreeRoot", "__init__", full_stack, str(type_), str(value_)
            )
            raise ex

    def read_sb_folder_structure(self):
        try:
            folders_to_process = [self]

            while len(folders_to_process) > 0:
                # get folder contents of next folder
                next_folder = folders_to_process.pop()
                child_ids = self.sb_access.get_child_ids(next_folder.sb_id)
                if child_ids is not None:
                    for child_id in child_ids:
                        # build child item and populate it
                        child_item = SBTreeNode(
                            next_folder.local_folder_path,
                            self.sb_access,
                            next_folder,
                            child_id,
                        )
                        # add child to folder/file tree
                        next_folder.folder_child_items[
                            child_item.sb_title
                        ] = child_item
                        # add child to list of possible folders to process
                        folders_to_process.append(child_item)
            # sort
            self._sort_items()
        except Exception as ex:
            # write uncaught exception to log and then raise it again
            type_, value_, traceback_ = sys.exc_info()
            full_stack = inspect.stack()
            ErrorLog.instance().write_log(
                "SBTreeRoot",
                "read_sb_folder_structure",
                full_stack,
                str(type_),
                str(value_),
            )
            raise ex

    def mirror_sciencebase_locally(
        self, copy_files=False, replace_local_copy=False
    ):
        try:
            if replace_local_copy and os.path.exists(self.local_folder_path):
                # clean up existing copy
                shutil.rmtree(self.local_folder_path)
            # mirror science base folders locally
            os.makedirs(self.local_folder_path, exist_ok=True)
            super().mirror_sciencebase_locally(copy_files)
        except Exception as ex:
            # write uncaught exception to log and then raise it again
            type_, value_, traceback_ = sys.exc_info()
            full_stack = inspect.stack()
            ErrorLog.instance().write_log(
                "SBTreeRoot",
                "mirror_sciencebase_locally",
                full_stack,
                str(type_),
                str(value_),
            )
            raise ex


if __name__ == "__main__":
    import shutil

    # clean test folders
    if os.path.exists("test_folder"):
        shutil.rmtree("test_folder")
    if os.path.exists("temp_test_folder"):
        shutil.rmtree("temp_test_folder")

    # test #1: mirror science base locally
    tree = SBTreeRoot(
        "test_folder",
        "spaulinski@usgs.gov",
        sb_root_folder_id="5fbe75fad34e4b9faad7e8a1",
    )
    tree.mirror_sciencebase_locally(True)

    # test #2: make changes to an existing file, test that it is correctly
    # marked as "sciencebase_out_of_date", upload to ScienceBase, verify
    # that files are marked as synced
    json_output = tree["data"]["j_output"]["json_output.txt"]
    file_status = json_output.file_status
    assert file_status == FileStatus.files_match
    # save a copy of file and modify
    shutil.copyfile(
        json_output.local_file_path,
        os.path.join(json_output.local_folder_path, "json_temp.txt"),
    )
    time.sleep(1)
    with open(json_output.local_file_path, "w") as fd:
        fd.write("!------- new file contents -------!")
    file_status = json_output.file_status
    assert file_status == FileStatus.sciencebase_out_of_date
    # upload modification to sciencebase
    json_output.upload_to_sciencebase()
    file_status = json_output.file_status
    assert file_status == FileStatus.files_match

    # test #3: create a folder and file locally, upload both to ScienceBase
    data = tree["data"]
    new_folder = os.path.join(data.local_folder_path, "added_data")
    os.mkdir(new_folder)
    new_file = os.path.join(new_folder, "new_data.txt")
    with open(new_file, "w") as fd:
        fd.write("new file data")
    # update the tree with local-only files and folders
    tree.populate_local_folder_structure()

    # upload new folder and file
    added_data = tree["data"]["added_data"]
    added_data.upload_to_sciencebase()
    new_data = added_data["new_data.txt"]
    new_data.upload_to_sciencebase()

    # download sciencebase to new tree to and test changes
    new_tree = SBTreeRoot(
        "temp_test_folder",
        "spaulinski@usgs.gov",
        sb_root_folder_id="5fbe75fad34e4b9faad7e8a1",
    )
    new_tree.mirror_sciencebase_locally(True)
    added_data = new_tree["data"]["added_data"]
    new_data = added_data["new_data.txt"]
    with open(new_data.local_file_path, "r") as fd:
        assert fd.readline() == "new file data"

    # test #4: change properties on existing sciencebase file and folder,
    # and commit changes to sciencebase

    # test #5:

    # restore sciencebase structure
    # test #1 restore
    # shutil.copyfile(os.path.join(json_output.local_folder_path,
    #                             'json_temp.txt'), json_output.local_file_path)
    # json_output.upload_to_sciencebase()
