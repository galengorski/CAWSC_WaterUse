import os, sys
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import seaborn as sns
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import outliers_detection as od


wu_df = pd.read_csv(r"C:\work\water_use\dataset\dailywu\pa\pa_master_training.csv")
wu_df = wu_df[wu_df['QUANTITY']>0]
wsid = wu_df['PUBLIC_WATER_SUPPLY'].unique()
all_vals = []
freq = []
with open('pa_spectral_graph.html', 'w') as fid:
    for ws in wsid:
        curr_ = wu_df[wu_df['PUBLIC_WATER_SUPPLY']==ws]
        curr_['REPORT_DATE'] = pd.to_datetime(curr_['REPORT_DATE'] )
        curr_['month'] = curr_['REPORT_DATE'].dt.month
        curr_ = curr_.sort_values(by = ['REPORT_DATE'])

        mm, mm_frac = od.get_monthly_frac(curr_, var_col='QUANTITY', date_col='REPORT_DATE', month_col='month')
        tmin, tmin_frac = od.get_monthly_frac(curr_, var_col='tmmn', date_col='REPORT_DATE', month_col='month')

        sign_sim, mag_sim = od.is_similair(mm_frac.values, tmin_frac.values)
        #norm_flow = curr_['QUANTITY']/curr_['QUANTITY'].sum()
        Mwu = curr_.groupby(by = ['month']).mean()

        f, Pxx_den = signal.periodogram(curr_['QUANTITY'].values, 1)
        f = f[1:]
        Pxx_den = Pxx_den[1:]
        #Pxx_den = Pxx_den[0:700]
        #f = f[0:700]
        #Pxx_den = Pxx_den / np.sum(Pxx_den)

        if 0:
            plt.semilogy(1/f, Pxx_den)
            plt.xlabel('Wavelength (days)')
            plt.ylabel('Power')
            plt.xscale('log')
        if 1:
            fig = go.Figure()
            fig = make_subplots(rows=1, cols=2)

            fig.add_trace(go.Scatter(x=(1/f), y=Pxx_den,
                                     mode='lines',
                                     name='lines'))
            fig.add_trace(go.Scatter(x = Mwu.index, y = Mwu['QUANTITY'], mode = 'lines', name = 'lines' ), row = 1, col = 2)

            fig.update_layout(title=ws)

            fig.update_yaxes(type="log", row=1, col=1)
            fig.update_xaxes(type="log", row=1, col = 1)
            fig.update_xaxes(title_text='Time', row=1)
            fig.update_yaxes(title_text='Power', col=1)
            fid.write(fig.to_html())
"""
fig.update_layout(title=ws,
        autosize = False,
        width = 800,
        height = 500)
"""



