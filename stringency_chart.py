#!/usr/bin/env python
# coding: utf-8

# Imports
import csv
from datetime import datetime
import io
from io import StringIO
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Ellipse, Polygon
import numpy as np
import os
import pandas as pd
from scipy.interpolate import make_interp_spline, BSpline, interp1d
import seaborn as sns
import sys

# Fonts
plt.rcParams['font.size'] = 42
plt.rcParams["font.weight"] = "normal"
plt.rcParams["axes.labelweight"] = "normal"
plt.rcParams["figure.titleweight"] = "bold"

# Colors
colors =  ["#3388ff","#A1E2E6", "#E6BDA1", "#B3A16B", "#678072", "#524A4A"]
cmap = cm.get_cmap('viridis', 5)
cmaplist = [cmap(i) for i in range(cmap.N)]

# Read data from "https://www.nature.com/articles/s41562-021-01079-8"
df = pd.read_csv('stringency_index.csv')
df = df.loc[df['country_code'].isin(['DNK','FIN','ISL','NOR','SWE'])]
df.reset_index(inplace=True,drop=True)
df = df.drop(columns=['Unnamed: 0','country_code'])

df2 = df.transpose()
header_row = 0
df2.columns = df2.iloc[header_row]
df2 = df2.reset_index(drop=False)
df2 = df2.rename(columns={'index':'date'})

df2 = df2[1:]
df2['date'] = df2['date'].apply(lambda x: datetime.strptime(x, "%d%b%Y"))
df2 = df2.set_index(df2['date'])
df2 = df2.loc["2020-03-01":"2021-02-28"]

from scipy.interpolate import make_interp_spline, BSpline

def smooth_line(dates,values):
	x = np.arange((len(dates)))
	xnew = np.linspace(x.min(),x.max(), 500)
	y = np.array(values)

	
	cub = interp1d(x,y, kind='cubic')
	y_cub = cub(xnew)
	spl = make_interp_spline(x,y,k=3)
	y_smooth = spl(xnew)
	#return xnew, y_smooth
	return xnew, y_cub

dk_x,dk_y = smooth_line(df2['date'],df2['Denmark'])
x = np.array((df2['date']))
datelist = pd.date_range("2020-01-01", periods=365).tolist()

# C7 Movement restrictions
c7 = pd.read_csv('c7_movementrestrictions.csv')
c7 = c7.loc[c7['country_code'].isin(['DNK','FIN','ISL','NOR','SWE'])]
c7.reset_index(inplace=True,drop=True)
c7 = c7.drop(columns=['Unnamed: 0','country_code'])
c7 = c7.transpose()
header_row = 0
c7.columns = c7.iloc[header_row]
c7 = c7.reset_index(drop=False)
c7 = c7.rename(columns={'index':'date'})

c7 = c7[1:]
c7['date'] = c7['date'].apply(lambda x: datetime.strptime(x, "%d%b%Y"))
c7 = c7.set_index(c7['date'])
c7 = c7.loc["2020-01-01":"2021-02-28"]

# C8 International Travel
c8 = pd.read_csv('c8_internationaltravel.csv')
c8 = c8.loc[c8['country_code'].isin(['DNK','FIN','ISL','NOR','SWE'])]
c8.reset_index(inplace=True,drop=True)
c8 = c8.drop(columns=['Unnamed: 0','country_code'])
c8 = c8.transpose()
header_row = 0
c8.columns = c8.iloc[header_row]
c8 = c8.reset_index(drop=False)
c8 = c8.rename(columns={'index':'date'})
c8 = c8[1:]
c8['date'] = c8['date'].apply(lambda x: datetime.strptime(x, "%d%b%Y"))
c8 = c8.set_index(c8['date'])
c8 = c8.loc["2020-01-01":"2021-02-28"]

c8 = c8.rename(columns={'Denmark':'DKC8','Finland':'FIC8','Iceland':'ISC8','Norway':'NOC8','Sweden':'SEC8'})

# Combine dataframes
combined = pd.concat([c7,c8], axis=1)
combined['DK_comb'] = combined['Denmark'] + combined['DKC8']
combined['FI_comb'] = combined['Finland'] + combined['FIC8']
combined['IS_comb'] = combined['Iceland'] + combined['ISC8']
combined['NO_comb'] = combined['Norway'] + combined['NOC8']
combined['SE_comb'] = combined['Sweden'] + combined['SEC8']
combined.reset_index(inplace=True, drop=True)
combined = combined[['DK_comb','FI_comb','IS_comb','NO_comb','SE_comb']]
combined = combined.rename(columns={'DK_comb':'Denmark','FI_comb':'Finland','IS_comb':'Iceland','NO_comb':'Norway','SE_comb':'Sweden'})
combined = combined.fillna(0)
combined = combined.transpose()

# Plot
f, ax = plt.subplots(ncols=1, figsize=(25,10))
heat = sns.heatmap(combined, cmap='OrRd', cbar_kws={'label': 'Stringency combined (c7 & c8)'}, xticklabels=True)
ax.set_ylabel(None)
plt.yticks(rotation=360)
x_dates = c8['date'].dt.strftime('%Y-%m-%d').sort_values().unique()
x_dates = x_dates[[0,25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,424]]
ax.set(xticks=([0,25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,424]))
ax.set_xticklabels(labels=x_dates, rotation=45, ha='right')
f.subplots_adjust(bottom=0.20)
plt.tight_layout()
plt.savefig(f'stringency_heat.JPEG',dpi=300)