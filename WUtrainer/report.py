import os, sys
import logging
import pandas as pd

class Logger():
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


