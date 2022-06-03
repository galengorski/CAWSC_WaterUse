import os, sys
import logging
import pandas as pd
import datetime
from art import *
from tabulate import tabulate
import json

class Logger():
    def __init__(self, filename, title = "WU", remove_log = True):
        self.filename = filename
        if remove_log:
            try:
                os.remove(filename)
            except:
                pass

        self.fidw = open(self.filename, 'w')
        title = text2art(title, "standard")
        self.fidw.write(title)
        self.fidw.write("\n\n")
        self.fidw.flush()

    def now(self):
        tim = datetime.datetime.now()
        str_time = tim.strftime("[%m/%d/%Y %H:%M:%S]:")
        return str_time

    def info(self, text, add_time = True):
        if add_time:
            msg = self.now() + text
        else:
            msg = text
        msg = msg + "\n"
        self.fidw.write(msg)
        self.fidw.flush()

    def to_table(self, df, title = '',  header = 5):
        self.fidw.write("\n")
        txt_table = tabulate(df.iloc[:header], headers='keys', tablefmt='psql')
        self.fidw.write(title + " -- Top {} lines --".format(header) + "\n")
        self.fidw.write(txt_table)
        self.fidw.write("\n")
        self.fidw.flush()

    def break_line(self):
        space = ""
        br = space + "=" * 25
        self.fidw.write("\n" + br + "\n")
        self.fidw.flush()

class Notebook():
    def __init__(self):
        self.header_comment = '# %%\n'


    def to_file(self, out_file):

        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(self.notebook, f, indent=2)

    def py2nb(self, py_str):
        # remove leading header comment
        header_comment = self.header_comment
        if py_str.startswith(header_comment):
            py_str = py_str[len(header_comment):]

        cells = []
        chunks = py_str.split('\n\n%s' % header_comment)

        for chunk in chunks:
            cell_type = 'code'
            if chunk.startswith("'''"):
                chunk = chunk.strip("'\n")
                cell_type = 'markdown'
            elif chunk.startswith('"""'):
                chunk = chunk.strip('"\n')
                cell_type = 'markdown'

            cell = {
                'cell_type': cell_type,
                'metadata': {},
                'source': chunk.splitlines(True),
            }

            if cell_type == 'code':
                cell.update({'outputs': [], 'execution_count': None})

            cells.append(cell)

        notebook = {
            'cells': cells,
            'metadata': {
                'anaconda-cloud': {},
                'kernelspec': {
                    'display_name': 'Python 3',
                    'language': 'python',
                    'name': 'python3'},
                'language_info': {
                    'codemirror_mode': {'name': 'ipython', 'version': 3},
                    'file_extension': '.py',
                    'mimetype': 'text/x-python',
                    'name': 'python',
                    'nbconvert_exporter': 'python',
                    'pygments_lexer': 'ipython3',
                    'version': '3.6.1'}},
            'nbformat': 4,
            'nbformat_minor': 4
        }
        self.notebook = notebook
        return notebook


class Logger2():
    def __init__(self, filename):
        self.filename = filename
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
        file_handler = logging.FileHandler(self.filename)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)


    def add_info_msg(self, msg):
        self.logger.info(msg)

    def df_info(self, df):
        columns = df.columns
        nrows = len(df)
        #self.logger.info("**********************")
        cols = ", ".join(columns)
        self.logger.info("Rows {}, \nNo. Features {}, \nFeatures names : {}".format(nrows, len(columns), cols))
        #self.logger.info("***********************")



