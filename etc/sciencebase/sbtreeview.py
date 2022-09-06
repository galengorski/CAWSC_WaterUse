import sys
import csv
import os
import inspect
from PyQt5.QtWidgets import QApplication, QMainWindow, QTreeView, QWidget
from PyQt5.QtWidgets import QFrame, QLineEdit, QPushButton, QTableView, QLabel
from PyQt5.QtWidgets import QCheckBox
from PyQt5.QtWidgets import QHeaderView, QInputDialog, QMessageBox
from PyQt5.Qt import QStandardItemModel, QStandardItem, QGridLayout
from PyQt5.QtGui import QFont, QColor, QBrush
from PyQt5.QtCore import QAbstractTableModel, Qt, QModelIndex
from sbsync import SBTreeRoot, FileStatus


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
            cls._instance._fd_log_file = open("debug_log_sbtreeview.txt", "w")

        return cls._instance

    def write_log(self, err_class, err_method, stack, err_type, value):
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


class CachedSettings:
    def __init__(self, settings_file="_sbtv_settings.txt"):
        self._settings_file = settings_file
        self.sb_root_folder_id = ""
        self.username = ""
        self.local_folder = ""

    def load_settings(self):
        try:
            self._log_dict = {}
            if os.path.isfile(self._settings_file):
                with open(self._settings_file, "r") as fd_read:
                    csv_rd = csv.reader(fd_read, delimiter=",", quotechar='"')
                    line_lst = next(csv_rd)
                    self.sb_root_folder_id = line_lst[0]
                    self.username = line_lst[1]
                    self.local_folder = line_lst[2]
        except Exception as ex:
            type_, value_, traceback_ = sys.exc_info()
            stack = inspect.stack()
            ErrorLog.instance().write_log(
                "CachedSettings",
                "load_settings",
                stack,
                str(type_),
                str(value_),
            )
            raise ex

    def update_settings(self, root_folder_id, username, local_folder):
        try:
            with open(self._settings_file, "w") as fd_write:
                fd_write.write(
                    "{},{},{}".format(root_folder_id, username, local_folder)
                )
        except Exception as ex:
            type_, value_, traceback_ = sys.exc_info()
            stack = inspect.stack()
            ErrorLog.instance().write_log(
                "CachedSettings",
                "update_settings",
                stack,
                str(type_),
                str(value_),
            )
            raise ex


class QHVLine(QFrame):
    def __init__(self, vertical=True):
        super(QHVLine, self).__init__()
        if vertical:
            self.setFrameShape(QFrame.VLine)
        else:
            self.setFrameShape(QFrame.HLine)

        self.setFrameShadow(QFrame.Sunken)
        self.setFixedHeight(20)


class TableModel(QAbstractTableModel):
    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return self._data[index.row()][index.column()]

    def rowCount(self, parent=QModelIndex()):
        return len(self._data)

    def columnCount(self, parent=QModelIndex()):
        return len(self._data[0])


class SbTreeViewItem(QStandardItem):
    def __init__(
        self,
        txt="",
        sb_tree_node=None,
        font_size=12,
        set_bold=False,
        color=QColor(0, 0, 0),
    ):
        try:
            super().__init__()

            fnt = QFont("Open Sans", font_size)
            fnt.setBold(set_bold)

            self.setEditable(False)
            self.setForeground(color)
            self.setFont(fnt)
            self.setText(txt)
            self.setData(sb_tree_node, Qt.UserRole)
        except Exception as ex:
            type_, value_, traceback_ = sys.exc_info()
            stack = inspect.stack()
            ErrorLog.instance().write_log(
                "SbTreeViewItem", "__init__", stack, str(type_), str(value_)
            )
            raise ex


class SBMainWindow(QMainWindow):
    def __init__(self):
        try:
            super().__init__()
            self.heading_style = (
                '<span style=" font-size:12pt; ' 'font-weight:600;">'
            )
            self.error_heading_style = (
                '<span style=" font-size:14pt; '
                'font-weight:600; color:#ff0000;">'
            )
            self.setWindowTitle("ScienceBase Viewer")
            self.resize(900, 700)
            self.password = ""
            self.file_item_list = []
            self.current_button_list = []
            self.checkbox = None
            self.bottom_display_text = None
            self.bottom_display_text_2 = None
            self.current_file = None
            self.tree = None
            self.clicked_tree_node = None

            # set up window grid
            self.central_widget = QWidget()
            self.setCentralWidget(self.central_widget)

            self.main_layout = QGridLayout()
            self.central_widget.setLayout(self.main_layout)

            self.top_layout = QGridLayout()
            self.main_layout.addLayout(self.top_layout, 1, 1)
            self.main_layout.addWidget(QHVLine(), 2, 1)
            self.bottom_layout = QGridLayout()
            self.main_layout.addLayout(self.bottom_layout, 3, 1)
            self.button_layout = QGridLayout()
            self.main_layout.addLayout(self.button_layout, 4, 1)

            # set up top layout with sb connect information
            self.label_root_folder_id = QLabel(
                "{}ScienceBase Root Folder "
                "ID<\\span"
                ">".format(self.heading_style)
            )
            self.top_layout.addWidget(self.label_root_folder_id, 1, 1)
            self.textbox_root_folder_id = QLineEdit(self)
            self.top_layout.addWidget(self.textbox_root_folder_id, 2, 1)

            self.label_username = QLabel(
                "{}Username <\\span" ">".format(self.heading_style)
            )
            self.top_layout.addWidget(self.label_username, 1, 2)
            self.textbox_username = QLineEdit(self)
            self.top_layout.addWidget(self.textbox_username, 2, 2)

            self.label_local_path = QLabel(
                "{}Path to Local Folder <\\span" ">".format(self.heading_style)
            )
            self.top_layout.addWidget(self.label_local_path, 1, 3)
            self.textbox_local_path = QLineEdit(self)
            self.top_layout.addWidget(self.textbox_local_path, 2, 3)

            self.connect = QPushButton(text="Connect")
            self.connect.clicked.connect(self.connect_event)
            self.top_layout.addWidget(self.connect, 2, 4)

            self.refresh = QPushButton(text="Refresh")
            self.refresh.clicked.connect(self.refresh_event)
            self.refresh.setEnabled(False)
            self.top_layout.addWidget(self.refresh, 2, 5)

            # set up file tree
            self.bottom_layout.setColumnMinimumWidth(3, 650)
            self.tree_view = QTreeView()
            self.bottom_layout.addWidget(self.tree_view, 1, 1)
            self.tree_view.setHeaderHidden(True)

            tree_model = QStandardItemModel()
            self.ui_root_node = tree_model.invisibleRootItem()

            # fill text for now
            cs = CachedSettings()
            cs.load_settings()
            self.textbox_root_folder_id.setText(cs.sb_root_folder_id)
            self.textbox_username.setText(cs.username)
            self.textbox_local_path.setText(cs.local_folder)
            self.tree_view.setModel(tree_model)
            self.tree_view.clicked.connect(self.treeview_click_event)

            # add list view
            self.bottom_layout.addWidget(QHVLine(False), 1, 2)
            self.table_view = QTableView()
            self.table_view.clicked.connect(self.tableview_click_event)
            self.table_view.horizontalHeader().setSectionResizeMode(
                QHeaderView.Stretch
            )
            self.table_model = QStandardItemModel()
            self._set_table_attributes()
            self.table_view.setModel(self.table_model)
            self.bottom_layout.addWidget(self.table_view, 1, 3)
        except Exception as ex:
            type_, value_, traceback_ = sys.exc_info()
            stack = inspect.stack()
            ErrorLog.instance().write_log(
                "SBMainWindow", "__init__", stack, str(type_), str(value_)
            )
            raise ex

    def _set_table_attributes(self):
        self.table_model.setHorizontalHeaderLabels(
            [
                "Filename",
                "Date Modified Local",
                "Date Modified SB",
                "Last Modified By",
            ]
        )
        self.table_view.setStyleSheet(
            "QHeaderView::section { background-color:gray }"
        )

    def connect_event(self):
        try:
            if len(self.textbox_root_folder_id.text()) == 0:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText(
                    "Can not connect without a ScienceBase root folder ID"
                )
                msg.setWindowTitle("ScienceBase Root Folder ID Needed")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                return
            ok = False
            if len(self.textbox_username.text()) > 0:
                self.password, ok = QInputDialog.getText(
                    self,
                    "Password",
                    "Please enter your ScienceBase password",
                    QLineEdit.Password,
                )

            if ok and self.password:
                # loading folder structure message
                self._set_message_text(
                    "Authenticating and downloading folder " "structure..."
                )

                # connect to ScienceBase
                self.tree = SBTreeRoot(
                    self.textbox_local_path.text(),
                    self.textbox_username.text(),
                    self.password,
                    self.textbox_root_folder_id.text(),
                )
                if not self.tree.authenticated:
                    self.tree = None
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText("Authentication to ScienceBase failed")
                    msg.setWindowTitle("Authentication to ScienceBase failed")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.exec_()
                    return

                self.refresh.setEnabled(True)

                # fill folder structure locally
                self.tree.mirror_sciencebase_locally(False)
                self.tree.populate_local_folder_structure()

                # loading folder structure message
                self._set_message_text("")

                # populate ui tree-view
                self.populate_ui_tree()

                # save settings
                cs = CachedSettings()
                cs.update_settings(
                    self.textbox_root_folder_id.text(),
                    self.textbox_username.text(),
                    self.textbox_local_path.text(),
                )
                if self.tree.sb_access.too_many_retries:
                    self.display_warning(
                        "WARNING: Too many retry attempts "
                        "while loading folder tree.  Folder "
                        "tree may be incomplete."
                    )
                    self.tree.sb_access.too_many_retries = False
        except Exception:
            type_, value_, traceback_ = sys.exc_info()
            stack = inspect.stack()
            ErrorLog.instance().write_log(
                "SBMainWindow", "connect_event", stack, str(type_), str(value_)
            )
            self.display_warning(
                "WARNING: Failed to populate folder tree. "
                "See error log for more information."
            )

    def display_warning(self, warning_msg):
        self._clear_labels_buttons()
        self.bottom_display_text = QLabel(
            "{}{}<\\span" ">".format(self.error_heading_style, warning_msg)
        )
        self.button_layout.addWidget(self.bottom_display_text, 1, 1)

    def refresh_event(self):
        try:
            # loading folder structure message
            self._set_message_text("Downloading folder structure...")
            self.tree.read_sb_folder_structure()
            self.tree.mirror_sciencebase_locally(False)
            self._set_message_text("")
            self.tree.populate_local_folder_structure()
            self.populate_ui_tree()
            # clear list view
            self.clicked_tree_node = None
            self._populate_list_view()
            if self.tree.sb_access.too_many_retries:
                self.display_warning(
                    "WARNING: Too many retry attempts"
                    "while refreshing folder tree.  Folder"
                    "tree may be incomplete."
                )
                self.tree.sb_access.too_many_retries = False
        except Exception:
            type_, value_, traceback_ = sys.exc_info()
            stack = inspect.stack()
            ErrorLog.instance().write_log(
                "SBMainWindow", "refresh_event", stack, str(type_), str(value_)
            )
            self._clear_labels_buttons()
            self.bottom_display_text = QLabel(
                "{}WARNING: Failed to refresh folder tree. "
                "See error log for more information. <\\span"
                ">".format(self.error_heading_style)
            )
            self.button_layout.addWidget(self.bottom_display_text, 1, 1)

    def treeview_click_event(self, val):
        try:
            self.clicked_tree_node = val.data(Qt.UserRole)
            download_all_btn, upload_all_btn = self._populate_list_view()
            if download_all_btn or upload_all_btn:
                self._add_labels_buttons(download_all_btn, upload_all_btn)
            else:
                self._clear_labels_buttons()
        except Exception:
            type_, value_, traceback_ = sys.exc_info()
            stack = inspect.stack()
            ErrorLog.instance().write_log(
                "SBMainWindow",
                "treeview_click_event",
                stack,
                str(type_),
                str(value_),
            )
            self._clear_labels_buttons()
            self.bottom_display_text = QLabel(
                "{}WARNING: An error occurred while handling the "
                "TreeView click event. "
                "See error log for more information. <\\span"
                ">".format(self.error_heading_style)
            )
            self.button_layout.addWidget(self.bottom_display_text, 1, 1)

    def _populate_list_view(self):
        try:
            # initialize
            self._clear_labels_buttons()
            self.current_file = None
            self.table_model = QStandardItemModel()
            self._set_table_attributes()
            vert_labels = []
            self.file_item_list = []
            file_to_download = False
            file_to_upload = False

            if self.clicked_tree_node is None:
                self.table_model.setVerticalHeaderLabels(vert_labels)
                self.table_view.setModel(self.table_model)
                return
            files_in_folder = self.clicked_tree_node.files
            can_download_all = len(files_in_folder) > 0
            can_upload_all = len(files_in_folder) > 0
            for child_file in files_in_folder.values():
                local_modify_time = child_file.local_modify_time
                if local_modify_time is None:
                    local_modify_str = ""
                    qsi_local_modify = QStandardItem(local_modify_str)
                    file_to_download = True
                else:
                    local_modify_str = local_modify_time.strftime(
                        "%m/%d/%Y, %H:%M:%S"
                    )
                    qsi_local_modify = QStandardItem(local_modify_str)
                    if (
                        child_file.file_status
                        == FileStatus.sciencebase_out_of_date
                        or child_file.file_status == FileStatus.merge_required
                    ):
                        qsi_local_modify.setBackground(QBrush(QColor("red")))
                        can_download_all = False
                    if (
                        child_file.file_status
                        == FileStatus.sciencebase_out_of_date
                    ):
                        file_to_upload = True
                sb_modify_time = child_file.sb_date_uploaded
                if sb_modify_time is None:
                    sb_modify_str = ""
                    qsi_sb_modify = QStandardItem(sb_modify_str)
                    file_to_upload = True
                else:
                    sb_modify_str = sb_modify_time.strftime(
                        "%m/%d/%Y, %H:%M:%S"
                    )
                    qsi_sb_modify = QStandardItem(sb_modify_str)
                    if (
                        child_file.file_status == FileStatus.local_out_of_date
                        or child_file.file_status == FileStatus.merge_required
                    ):
                        qsi_sb_modify.setBackground(QBrush(QColor("red")))
                        can_upload_all = False
                    if child_file.file_status == FileStatus.local_out_of_date:
                        file_to_download = True

                if child_file.sb_uploaded_by is None:
                    sb_uploaded_by_str = ""
                else:
                    sb_uploaded_by_str = child_file.sb_uploaded_by
                self.table_model.appendRow(
                    [
                        QStandardItem(child_file.sb_name),
                        qsi_local_modify,
                        qsi_sb_modify,
                        QStandardItem(sb_uploaded_by_str),
                    ]
                )
                vert_labels.append("")
                self.file_item_list.append(child_file)
            self.table_model.setVerticalHeaderLabels(vert_labels)
            self.table_view.setModel(self.table_model)
            return (
                can_download_all and file_to_download,
                can_upload_all and file_to_upload,
            )
        except Exception:
            type_, value_, traceback_ = sys.exc_info()
            stack = inspect.stack()
            ErrorLog.instance().write_log(
                "SBMainWindow",
                "_populate_list_view",
                stack,
                str(type_),
                str(value_),
            )
            self._clear_labels_buttons()
            self.bottom_display_text = QLabel(
                "{}WARNING: An error occurred while populating the list"
                "view. See error log for more information. <\\span"
                ">".format(self.error_heading_style)
            )
            self.button_layout.addWidget(self.bottom_display_text, 1, 1)

    def tableview_click_event(self, val):
        try:
            self._clear_labels_buttons()
            self.current_file = None

            # add new labels and buttons
            self.current_file = self.file_item_list[val.row()]
            self._add_labels_buttons()
        except Exception:
            type_, value_, traceback_ = sys.exc_info()
            stack = inspect.stack()
            ErrorLog.instance().write_log(
                "SBMainWindow",
                "tableview_click_event",
                stack,
                str(type_),
                str(value_),
            )
            self._clear_labels_buttons()
            self.bottom_display_text = QLabel(
                "{}WARNING: An error occurred while handling the "
                "TableView click event. "
                "See error log for more information. <\\span"
                ">".format(self.error_heading_style)
            )
            self.button_layout.addWidget(self.bottom_display_text, 1, 1)

    def download_event(self, val):
        try:
            if self.current_file is not None:
                # display downloading message
                self._set_message_text(
                    'Downloading file "{}"..'
                    ".".format(self.current_file.sb_name)
                )
                # download file
                if self.current_file.download_file_from_sciencebase():
                    self._clear_labels_buttons()
                    self._add_labels_buttons()
                    # update file modification date in list view
                    if self.clicked_tree_node is not None:
                        self._populate_list_view()
                    # display success message
                    self._clear_labels_buttons()
                    self.bottom_display_text = QLabel(
                        "{}File successfully downloaded <\\span"
                        ">".format(self.heading_style)
                    )
                    self.button_layout.addWidget(
                        self.bottom_display_text, 1, 1
                    )
        except Exception:
            type_, value_, traceback_ = sys.exc_info()
            stack = inspect.stack()
            ErrorLog.instance().write_log(
                "SBMainWindow",
                "download_event",
                stack,
                str(type_),
                str(value_),
            )
            self._clear_labels_buttons()
            self.bottom_display_text = QLabel(
                "{}WARNING: An error occurred while handling the "
                "download event. "
                "See error log for more information. <\\span"
                ">".format(self.error_heading_style)
            )
            self.button_layout.addWidget(self.bottom_display_text, 1, 1)

    def download_fd_event(self, val):
        try:
            if self.clicked_tree_node is not None:
                files_downloaded = []
                files_failed = []
                for child_file in self.clicked_tree_node.files.values():
                    if (
                        child_file.file_status == FileStatus.local_file_missing
                        or child_file.file_status
                        == FileStatus.local_out_of_date
                    ):
                        # display downloading message
                        self._set_message_text(
                            'Downloading file "{}"..'
                            ".".format(child_file.sb_name)
                        )
                        # download file
                        if child_file.download_file_from_sciencebase():
                            files_downloaded.append(child_file.sb_name)
                        else:
                            files_failed.append(child_file.sb_name)
                # update modification dates
                self._populate_list_view()
                # display results message
                self._clear_labels_buttons()
                if files_downloaded:
                    sfs = ", ".join(files_downloaded)
                    success_message = (
                        "Files successfully downloaded" ": {}".format(sfs)
                    )
                    self.bottom_display_text = QLabel(
                        "{}{} <\\span>".format(
                            self.heading_style, success_message
                        )
                    )
                    self.button_layout.addWidget(
                        self.bottom_display_text, 1, 1
                    )
                if files_failed:
                    ffs = ", ".join(files_failed)
                    failed_message = "Files failed to download" ": {}".format(
                        ffs
                    )
                    self.bottom_display_text_2 = QLabel(
                        "{}{} <\\span>".format(
                            self.heading_style, failed_message
                        )
                    )
                    self.button_layout.addWidget(
                        self.bottom_display_text_2, 1, 2
                    )
        except Exception:
            type_, value_, traceback_ = sys.exc_info()
            stack = inspect.stack()
            ErrorLog.instance().write_log(
                "SBMainWindow",
                "download_fd_event",
                stack,
                str(type_),
                str(value_),
            )
            self._clear_labels_buttons()
            self.bottom_display_text = QLabel(
                "{}WARNING: An error occurred while handling the "
                "download folder event. "
                "See error log for more information. <\\span"
                ">".format(self.error_heading_style)
            )
            self.button_layout.addWidget(self.bottom_display_text, 1, 1)

    def upload_event(self, val):
        try:
            if self.current_file is not None:
                convert_zip = (
                    self.checkbox is not None and self.checkbox.isChecked()
                )
                # display uploading message
                self._set_message_text(
                    'Uploading file "{}"..'
                    ".".format(self.current_file.sb_name)
                )
                # upload file
                if self.current_file.upload_to_sciencebase(
                    convert_to_zip=convert_zip
                ):
                    self._clear_labels_buttons()
                    self._add_labels_buttons()
                    # update file modification date in list view
                    if self.clicked_tree_node is not None:
                        self._populate_list_view()
                    # display success message
                    self._clear_labels_buttons()
                    self.bottom_display_text = QLabel(
                        "{}File successfully uploaded <\\span"
                        ">".format(self.heading_style)
                    )
                    self.button_layout.addWidget(
                        self.bottom_display_text, 1, 1
                    )
        except Exception:
            type_, value_, traceback_ = sys.exc_info()
            stack = inspect.stack()
            ErrorLog.instance().write_log(
                "SBMainWindow", "upload_event", stack, str(type_), str(value_)
            )
            self._clear_labels_buttons()
            self.bottom_display_text = QLabel(
                "{}WARNING: An error occurred while handling the "
                "upload event. "
                "See error log for more information. <\\span"
                ">".format(self.error_heading_style)
            )
            self.button_layout.addWidget(self.bottom_display_text, 1, 1)

    def upload_fd_event(self, val):
        try:
            if self.clicked_tree_node is not None:
                files_uploaded = []
                files_failed = []
                # upload files
                for child_file in self.clicked_tree_node.files.values():
                    if (
                        child_file.file_status
                        == FileStatus.sciencebase_file_missing
                        or child_file.file_status
                        == FileStatus.sciencebase_out_of_date
                    ):
                        # display uploading message
                        self._set_message_text(
                            'Uploading file "{}"..'
                            ".".format(child_file.sb_name)
                        )
                        # upload file
                        if child_file.upload_to_sciencebase():
                            files_uploaded.append(child_file.sb_name)
                        else:
                            files_failed.append(child_file.sb_name)
                # update modification dates
                self._populate_list_view()
                # display results message
                self._clear_labels_buttons()
                if files_uploaded:
                    sfs = ", ".join(files_uploaded)
                    success_message = (
                        "Files successfully uploaded" ": {}".format(sfs)
                    )
                    self.bottom_display_text = QLabel(
                        "{}{} <\\span>".format(
                            self.heading_style, success_message
                        )
                    )
                    self.button_layout.addWidget(
                        self.bottom_display_text, 1, 1
                    )
                if files_failed:
                    ffs = ", ".join(files_failed)
                    failed_message = "Files failed to upload" ": {}".format(
                        ffs
                    )
                    self.bottom_display_text_2 = QLabel(
                        "{}{} <\\span>".format(
                            self.heading_style, failed_message
                        )
                    )
                    self.button_layout.addWidget(
                        self.bottom_display_text_2, 1, 2
                    )
        except Exception:
            type_, value_, traceback_ = sys.exc_info()
            stack = inspect.stack()
            ErrorLog.instance().write_log(
                "SBMainWindow",
                "upload_fd_event",
                stack,
                str(type_),
                str(value_),
            )
            self._clear_labels_buttons()
            self.bottom_display_text = QLabel(
                "{}WARNING: An error occurred while handling the "
                "upload folder event. "
                "See error log for more information. <\\span"
                ">".format(self.error_heading_style)
            )
            self.button_layout.addWidget(self.bottom_display_text, 1, 1)

    def populate_ui_tree(self):
        try:
            # clear tree
            self.ui_root_node.removeRows(0, self.ui_root_node.rowCount())

            # populate root node
            ui_top_folder = SbTreeViewItem(self.tree.sb_title, self.tree, 14)
            self.ui_root_node.appendRow(ui_top_folder)
            self.recursive_populate_ui_tree(self.tree, ui_top_folder)
            self.tree_view.expandAll()
        except Exception as ex:
            type_, value_, traceback_ = sys.exc_info()
            stack = inspect.stack()
            ErrorLog.instance().write_log(
                "SBMainWindow",
                "populate_ui_tree",
                stack,
                str(type_),
                str(value_),
            )
            raise ex

    def recursive_populate_ui_tree(self, tree, ui_parent_node):
        try:
            for child in tree.folder_child_items.values():
                ui_child = SbTreeViewItem(child.sb_title, child, 14)
                ui_parent_node.appendRow(ui_child)
                if child.sb_has_children or child.local_has_children:
                    self.recursive_populate_ui_tree(child, ui_child)
        except Exception as ex:
            type_, value_, traceback_ = sys.exc_info()
            stack = inspect.stack()
            ErrorLog.instance().write_log(
                "SBMainWindow",
                "recursive_populate_ui_tree",
                stack,
                str(type_),
                str(value_),
            )
            raise ex

    def _set_message_text(self, message):
        # display uploading message
        self._clear_labels_buttons()
        self.bottom_display_text = QLabel(
            "{}{} <\\span" ">".format(self.heading_style, message)
        )
        self.button_layout.addWidget(self.bottom_display_text, 1, 1)
        QApplication.processEvents()

    def _add_labels_buttons(
        self, download_all_btn=False, upload_all_btn=False
    ):
        try:
            if download_all_btn:
                self.bottom_display_text = QLabel(
                    "{}All ScienceBase files are either more recent "
                    "or up to date with local files. <\\span"
                    ">".format(self.heading_style)
                )
                self.button_layout.addWidget(self.bottom_display_text, 1, 1)
                download_sfd = QPushButton(
                    text="Download ScienceBase Folder "
                    "Contents To Local Machine"
                )
                download_sfd.clicked.connect(self.download_fd_event)
                self.button_layout.addWidget(download_sfd, 2, 1)
                self.current_button_list.append(download_sfd)
                return
            elif upload_all_btn:
                self.bottom_display_text = QLabel(
                    "{}All local files are either more recent "
                    "or up to date with ScienceBase files. <\\span"
                    ">".format(self.heading_style)
                )
                self.button_layout.addWidget(self.bottom_display_text, 1, 1)
                upload_lfd = QPushButton(
                    text="Upload Local Folder " "Contents To ScienceBase"
                )
                upload_lfd.clicked.connect(self.upload_fd_event)
                self.button_layout.addWidget(upload_lfd, 2, 1)
                self.current_button_list.append(upload_lfd)
                return
            if self.current_file.file_status == FileStatus.files_match:
                self.bottom_display_text = QLabel(
                    "{}Local and ScienceBase files match <\\span"
                    ">".format(self.heading_style)
                )
                self.button_layout.addWidget(self.bottom_display_text, 1, 1)
            elif (
                self.current_file.file_status == FileStatus.local_file_missing
            ):
                self.bottom_display_text = QLabel(
                    "{}Local file does not exist <\\span"
                    ">".format(self.heading_style)
                )
                self.button_layout.addWidget(self.bottom_display_text, 1, 1)
                download = QPushButton(text="Download File From ScienceBase")
                download.clicked.connect(self.download_event)
                self.button_layout.addWidget(download, 2, 1)
                self.current_button_list.append(download)
            elif (
                self.current_file.file_status
                == FileStatus.sciencebase_file_missing
            ):
                self.bottom_display_text = QLabel(
                    "{}ScienceBase file does not exist <\\span"
                    ">".format(self.heading_style)
                )
                self.button_layout.addWidget(self.bottom_display_text, 1, 1)
                upload = QPushButton(text="Upload File To ScienceBase")
                upload.clicked.connect(self.upload_event)
                self.button_layout.addWidget(upload, 2, 1)
                self.current_button_list.append(upload)
                self.checkbox = QCheckBox(text="Create zip file")
                self.button_layout.addWidget(self.checkbox, 3, 1)
            elif self.current_file.file_status == FileStatus.local_out_of_date:
                self.bottom_display_text = QLabel(
                    "{}Local file is out of date <\\span"
                    ">".format(self.heading_style)
                )
                self.button_layout.addWidget(self.bottom_display_text, 1, 1)
                download = QPushButton(
                    text="Update Local File " "(ScienceBase-->Local)"
                )
                download.clicked.connect(self.download_event)
                self.button_layout.addWidget(download, 2, 1)
                self.current_button_list.append(download)
            elif (
                self.current_file.file_status
                == FileStatus.sciencebase_out_of_date
            ):
                self.bottom_display_text = QLabel(
                    "{}ScienceBase file is out of date <\\span"
                    ">".format(self.heading_style)
                )
                self.button_layout.addWidget(self.bottom_display_text, 1, 1)
                upload = QPushButton(
                    text="Update ScienceBase File " "(Local-->ScienceBase)"
                )
                upload.clicked.connect(self.upload_event)
                self.button_layout.addWidget(upload, 2, 1)
                self.current_button_list.append(upload)
            elif self.current_file.file_status == FileStatus.merge_required:
                self.bottom_display_text = QLabel(
                    "{}ScienceBase and local files are both out of date "
                    "<\\span>".format(self.heading_style)
                )
                self.button_layout.addWidget(self.bottom_display_text, 1, 1)
                upload = QPushButton(
                    text="Update ScienceBase File " "(Local-->ScienceBase"
                )
                upload.setStyleSheet("background-color: red")
                upload.clicked.connect(self.upload_event)
                self.button_layout.addWidget(upload, 2, 1)
                self.current_button_list.append(upload)
                download = QPushButton(
                    text="Update Local File " "(ScienceBase-->Local)"
                )
                download.setStyleSheet("background-color: red")
                download.clicked.connect(self.download_event)
                self.button_layout.addWidget(download, 2, 2)
                self.current_button_list.append(download)
            elif (
                self.current_file.file_status
                == FileStatus.out_of_date_merge_status_unknown
            ):
                self.bottom_display_text = QLabel(
                    "{}ScienceBase and local files exist but "
                    "synchronization information missing."
                    "<\\span>".format(self.heading_style)
                )
                self.button_layout.addWidget(self.bottom_display_text, 1, 1)
                upload = QPushButton(
                    text="Update ScienceBase File " "(Local-->ScienceBase"
                )
                upload.setStyleSheet("background-color: red")
                upload.clicked.connect(self.upload_event)
                self.button_layout.addWidget(upload, 2, 1)
                self.current_button_list.append(upload)
                download = QPushButton(
                    text="Update Local File " "(ScienceBase-->Local)"
                )
                download.setStyleSheet("background-color: red")
                download.clicked.connect(self.download_event)
                self.button_layout.addWidget(download, 2, 2)
                self.current_button_list.append(download)
        except Exception as ex:
            type_, value_, traceback_ = sys.exc_info()
            stack = inspect.stack()
            ErrorLog.instance().write_log(
                "SBMainWindow",
                "_add_labels_buttons",
                stack,
                str(type_),
                str(value_),
            )
            raise ex

    def _clear_labels_buttons(self):
        try:
            # clean up old labels and buttons
            for button in self.current_button_list:
                button.setParent(None)
            if self.bottom_display_text is not None:
                self.bottom_display_text.setParent(None)
            if self.bottom_display_text_2 is not None:
                self.bottom_display_text_2.setParent(None)
            if self.checkbox is not None:
                self.checkbox.setParent(None)
                self.checkbox = None
            self.current_button_list = []
        except Exception as ex:
            type_, value_, traceback_ = sys.exc_info()
            stack = inspect.stack()
            ErrorLog.instance().write_log(
                "SBMainWindow",
                "_clear_labels_buttons",
                stack,
                str(type_),
                str(value_),
            )
            raise ex


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = SBMainWindow()
    main_window.show()
    sys.exit(app.exec_())
