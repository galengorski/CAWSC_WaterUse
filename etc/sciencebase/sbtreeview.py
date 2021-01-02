import sys
import csv
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QTreeView, QWidget
from PyQt5.QtWidgets import QFrame, QLineEdit, QPushButton, QTableView, QLabel
from PyQt5.QtWidgets import QHeaderView, QInputDialog, QMessageBox
from PyQt5.Qt import QStandardItemModel, QStandardItem, QGridLayout
from PyQt5.QtGui import QFont, QColor, QBrush
from PyQt5.QtCore import QAbstractTableModel, Qt, QModelIndex
from sbsync import SBTreeRoot, FileStatus


class CachedSettings:
    def __init__(self, settings_file='_sbtv_settings.txt'):
        self._settings_file = settings_file
        self.sb_root_folder_id = ''
        self.username = ''
        self.local_folder = ''

    def load_settings(self):
        self._log_dict = {}
        if os.path.isfile(self._settings_file):
            with open(self._settings_file, 'r') as fd_read:
                csv_rd = csv.reader(fd_read, delimiter=',', quotechar='\"')
                line_lst = next(csv_rd)
                self.sb_root_folder_id = line_lst[0]
                self.username = line_lst[1]
                self.local_folder = line_lst[2]

    def update_settings(self, root_folder_id, username, local_folder):
        with open(self._settings_file, 'w') as fd_write:
            fd_write.write('{},{},{}'.format(root_folder_id, username,
                                             local_folder))


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
    def __init__(self, txt='', sb_tree_node=None, font_size=12, set_bold=False,
                 color=QColor(0, 0, 0)):
        super().__init__()

        fnt = QFont('Open Sans', font_size)
        fnt.setBold(set_bold)

        self.setEditable(False)
        self.setForeground(color)
        self.setFont(fnt)
        self.setText(txt)
        self.setData(sb_tree_node, Qt.UserRole)


class SBMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.heading_style = '<span style=" font-size:12pt; font-weight:600;">'
        self.setWindowTitle('ScienceBase Viewer')
        self.resize(900, 700)
        self.password = ''
        self.file_item_list = []
        self.current_button_list = []
        self.bottom_display_text = None
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
        self.label_root_folder_id = QLabel('{}ScienceBase Root Folder ID<\\span'
                                           '>'.format(self.heading_style))
        self.top_layout.addWidget(self.label_root_folder_id, 1, 1)
        self.textbox_root_folder_id = QLineEdit(self)
        self.top_layout.addWidget(self.textbox_root_folder_id, 2, 1)

        self.label_username = QLabel('{}Username <\\span'
                                     '>'.format(self.heading_style))
        self.top_layout.addWidget(self.label_username, 1, 2)
        self.textbox_username = QLineEdit(self)
        self.top_layout.addWidget(self.textbox_username, 2, 2)

        self.label_local_path = QLabel('{}Path to Local Folder <\\span'
                                       '>'.format(self.heading_style))
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
            QHeaderView.Stretch)
        self.table_model = QStandardItemModel()
        self._set_table_attributes()
        self.table_view.setModel(self.table_model)
        self.bottom_layout.addWidget(self.table_view, 1, 3)

    def _set_table_attributes(self):
        self.table_model.setHorizontalHeaderLabels(['Filename',
                                                    'Date Modified Local',
                                                    'Date Modified SB',
                                                    'Last Modified By'])
        self.table_view.setStyleSheet(
            "QHeaderView::section { background-color:gray; font: bold 15px; }")

    def connect_event(self):
        if len(self.textbox_root_folder_id.text()) == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Can not connect without a ScienceBase root folder ID")
            msg.setWindowTitle("ScienceBase Root Folder ID Needed")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return
        ok = False
        if len(self.textbox_username.text()) > 0:
            self.password, ok = QInputDialog.getText(
                    self, "Password", "Please enter your ScienceBase password",
                    QLineEdit.Password)

        if ok and self.password:
            # connect to ScienceBase
            self.tree = SBTreeRoot(self.textbox_local_path.text(),
                                   self.textbox_username.text(),
                                   self.password,
                                   self.textbox_root_folder_id.text())
            if not self.tree.authenticated:
                self.tree = None
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText(
                    "Authentication to ScienceBase failed")
                msg.setWindowTitle("Authentication to ScienceBase failed")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                return

            self.refresh.setEnabled(True)

            # fill folder structure locally
            self.tree.mirror_sciencebase_locally(False)
            self.tree.populate_local_folder_structure()

            # populate ui tree-view
            self.populate_ui_tree()

            # save settings
            cs = CachedSettings()
            cs.update_settings(self.textbox_root_folder_id.text(),
                               self.textbox_username.text(),
                               self.textbox_local_path.text())

    def refresh_event(self):
        self.tree.read_sb_folder_structure()
        self.tree.mirror_sciencebase_locally(False)
        self.tree.populate_local_folder_structure()
        self.populate_ui_tree()
        # clear list view
        self.clicked_tree_node = None
        self._populate_list_view()

    def treeview_click_event(self, val):
        self.clicked_tree_node = val.data(Qt.UserRole)
        self._populate_list_view()

    def _populate_list_view(self):
        self._clear_labels_buttons()
        self.current_file = None
        self.table_model = QStandardItemModel()
        self._set_table_attributes()
        vert_labels = []
        self.file_item_list = []
        if self.clicked_tree_node is None:
            self.table_model.setVerticalHeaderLabels(vert_labels)
            self.table_view.setModel(self.table_model)
            return
        for child_file in self.clicked_tree_node.files.values():
            local_modify_time = child_file.local_modify_time
            if local_modify_time is None:
                local_modify_str = ''
                qsi_local_modify = QStandardItem(local_modify_str)
            else:
                local_modify_str = local_modify_time.strftime(
                    "%m/%d/%Y, %H:%M:%S")
                qsi_local_modify = QStandardItem(local_modify_str)
                if child_file.file_status == \
                        FileStatus.sciencebase_out_of_date or \
                        child_file.file_status == FileStatus.merge_required:
                    qsi_local_modify.setBackground(QBrush(QColor("red")))
            sb_modify_time = child_file.sb_date_uploaded
            if sb_modify_time is None:
                sb_modify_str = ''
                qsi_sb_modify = QStandardItem(sb_modify_str)
            else:
                sb_modify_str = sb_modify_time.strftime(
                    "%m/%d/%Y, %H:%M:%S")
                qsi_sb_modify = QStandardItem(sb_modify_str)
                if child_file.file_status == \
                        FileStatus.local_out_of_date or \
                        child_file.file_status == FileStatus.merge_required:
                    qsi_sb_modify.setBackground(QBrush(QColor("red")))

            if child_file.sb_uploaded_by is None:
                sb_uploaded_by_str = ''
            else:
                sb_uploaded_by_str = child_file.sb_uploaded_by
            self.table_model.appendRow([QStandardItem(child_file.sb_name),
                                        qsi_local_modify,
                                        qsi_sb_modify,
                                        QStandardItem(sb_uploaded_by_str)])
            vert_labels.append('')
            self.file_item_list.append(child_file)
        self.table_model.setVerticalHeaderLabels(vert_labels)
        self.table_view.setModel(self.table_model)

    def tableview_click_event(self, val):
        self._clear_labels_buttons()
        self.current_file = None

        # add new labels and buttons
        self.current_file = self.file_item_list[val.row()]
        self._add_labels_buttons()

    def download_event(self, val):
        if self.current_file is not None:
            if self.current_file.download_file_from_sciencebase():
                self._clear_labels_buttons()
                self._add_labels_buttons()
                # update file modification date in list view
                if self.clicked_tree_node is not None:
                    self._populate_list_view()
                # display success message
                self._clear_labels_buttons()
                self.bottom_display_text = \
                    QLabel('{}File successfully downloaded <\\span'
                           '>'.format(self.heading_style))
                self.button_layout.addWidget(self.bottom_display_text, 1, 1)

    def upload_event(self, val):
        if self.current_file is not None:
            if self.current_file.upload_to_sciencebase():
                self._clear_labels_buttons()
                self._add_labels_buttons()
                # update file modification date in list view
                if self.clicked_tree_node is not None:
                    self._populate_list_view()
                # display success message
                self._clear_labels_buttons()
                self.bottom_display_text = \
                    QLabel('{}File successfully uploaded <\\span'
                           '>'.format(self.heading_style))
                self.button_layout.addWidget(self.bottom_display_text, 1, 1)

    def populate_ui_tree(self):
        # clear tree
        self.ui_root_node.removeRows(0, self.ui_root_node.rowCount())

        # populate root node
        ui_top_folder = SbTreeViewItem(self.tree.sb_title, self.tree, 14)
        self.ui_root_node.appendRow(ui_top_folder)
        self.recursive_populate_ui_tree(self.tree, ui_top_folder)
        self.tree_view.expandAll()

    def recursive_populate_ui_tree(self, tree, ui_parent_node):
        for child in tree.folder_child_items.values():
            ui_child = SbTreeViewItem(child.sb_title, child, 14)
            ui_parent_node.appendRow(ui_child)
            if child.sb_has_children:
                self.recursive_populate_ui_tree(child, ui_child)

    def _add_labels_buttons(self):
        if self.current_file.file_status == FileStatus.files_match:
            self.bottom_display_text = \
                QLabel('{}Local and ScienceBase files match <\\span'
                       '>'.format(self.heading_style))
            self.button_layout.addWidget(self.bottom_display_text, 1, 1)
        elif self.current_file.file_status == FileStatus.local_file_missing:
            self.bottom_display_text = \
                QLabel('{}Local file does not exist <\\span'
                       '>'.format(self.heading_style))
            self.button_layout.addWidget(self.bottom_display_text, 1, 1)
            download = QPushButton(text="Download File From ScienceBase")
            download.clicked.connect(self.download_event)
            self.button_layout.addWidget(download, 2, 1)
            self.current_button_list.append(download)
        elif self.current_file.file_status == \
                FileStatus.sciencebase_file_missing:
            self.bottom_display_text = \
                QLabel('{}ScienceBase file does not exist <\\span'
                       '>'.format(self.heading_style))
            self.button_layout.addWidget(self.bottom_display_text, 1, 1)
            upload = QPushButton(text="Upload File To ScienceBase")
            upload.clicked.connect(self.upload_event)
            self.button_layout.addWidget(upload, 2, 1)
            self.current_button_list.append(upload)
        elif self.current_file.file_status == FileStatus.local_out_of_date:
            self.bottom_display_text = \
                QLabel('{}Local file is out of date <\\span'
                       '>'.format(self.heading_style))
            self.button_layout.addWidget(self.bottom_display_text, 1, 1)
            download = QPushButton(text="Update Local File "
                                        "(ScienceBase-->Local)")
            download.clicked.connect(self.download_event)
            self.button_layout.addWidget(download, 2, 1)
            self.current_button_list.append(download)
        elif self.current_file.file_status == \
                FileStatus.sciencebase_out_of_date:
            self.bottom_display_text = \
                QLabel('{}ScienceBase file is out of date <\\span'
                       '>'.format(self.heading_style))
            self.button_layout.addWidget(self.bottom_display_text, 1, 1)
            upload = QPushButton(text="Update ScienceBase File "
                                      "(Local-->ScienceBase)")
            upload.clicked.connect(self.upload_event)
            self.button_layout.addWidget(upload, 2, 1)
            self.current_button_list.append(upload)
        elif self.current_file.file_status == \
                FileStatus.merge_required:
            self.bottom_display_text = \
                QLabel('{}ScienceBase and local files are both out of date '
                       '<\\span>'.format(self.heading_style))
            self.button_layout.addWidget(self.bottom_display_text, 1, 1)
            upload = QPushButton(text="Update ScienceBase File "
                                      "(Local-->ScienceBase")
            upload.setStyleSheet("background-color: red")
            upload.clicked.connect(self.upload_event)
            self.button_layout.addWidget(upload, 2, 1)
            self.current_button_list.append(upload)
            download = QPushButton(text="Update Local File "
                                        "(ScienceBase-->Local)")
            download.setStyleSheet("background-color: red")
            download.clicked.connect(self.download_event)
            self.button_layout.addWidget(download, 2, 2)
            self.current_button_list.append(download)
        elif self.current_file.file_status == \
                FileStatus.out_of_date_merge_status_unknown:
            self.bottom_display_text = \
                QLabel('{}ScienceBase and local files exist but synchronization '
                       'information missing.'
                       '<\\span>'.format(self.heading_style))
            self.button_layout.addWidget(self.bottom_display_text, 1, 1)
            upload = QPushButton(text="Update ScienceBase File "
                                      "(Local-->ScienceBase")
            upload.setStyleSheet("background-color: red")
            upload.clicked.connect(self.upload_event)
            self.button_layout.addWidget(upload, 2, 1)
            self.current_button_list.append(upload)
            download = QPushButton(text="Update Local File "
                                        "(ScienceBase-->Local)")
            download.setStyleSheet("background-color: red")
            download.clicked.connect(self.download_event)
            self.button_layout.addWidget(download, 2, 2)
            self.current_button_list.append(download)

    def _clear_labels_buttons(self):
        # clean up old labels and buttons
        for button in self.current_button_list:
            #self.button_layout.removeWidget(button)
            button.setParent(None)
        if self.bottom_display_text is not None:
            #self.button_layout.removeWidget(self.bottom_display_text)
            self.bottom_display_text.setParent(None)
        self.current_button_list = []


app = QApplication(sys.argv)
main_window = SBMainWindow()
main_window.show()
sys.exit(app.exec_())
