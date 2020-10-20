import plotly as py
import pandas as pd
import numpy as np

from datetime import datetime
from datetime import time as dt_tm
from datetime import date as dt_date

import plotly.plotly as py
import plotly.tools as plotly_tools
import plotly.graph_objs as go



class Hreport():
    def __init__(self):
        self.filename  = "wu.html"
        self.sections = []
        pass

    def add_section(self, section):
        """
        usually this is called Div in html; the div can contain
            - break line
            - title or section name
            - regular text (multiple paragraphs)
            - tables
            - figures
        :return:
        """
        self.sections.append(section)

    def add_break_line(self, section, options = {}):
        pass
    def add_text(self, section, text, options = {}):
        pass
    def add_figure(self, section, figObj, options = {}):
        pass
    def add_table(self, section, df, options = {}):
        pass
    def from_html(self, fname):
        pass
    def to_html(self, fname):
        pass

    