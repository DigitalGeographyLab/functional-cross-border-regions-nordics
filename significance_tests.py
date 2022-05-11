#!/usr/bin/env python
# coding: utf-8

# Imports
import pandas as pd
import psycopg2
import psycopg2.extras as extras
import operator
import numpy as np
import geopandas as gpd
import json
from sqlalchemy import create_engine, func, distinct
import io
from io import StringIO
import tempfile
import os
import csv
#from pyproj import CRS
from shapely.geometry import Point, LineString, Polygon
import geojson
import folium
#import osmnx as ox
import contextily as ctx
from shapely import wkt
import sys
from scipy.stats import shapiro
from datetime import datetime
import db_connection as db_con
starttime = datetime.now()
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu
from scipy.stats import kruskal
import scikit_posthocs as sp

# Import credential and connection settings
db_name = db_con.db_name
db_username = db_con.db_username
db_host = db_con.db_host
db_port = db_con.db_port
db_pwd = db_con.db_pwd
engine_string = f"postgresql://{db_username}:{db_pwd}@{db_host}:{db_port}/{db_name}"
db_engine = create_engine(engine_string)

# Weighting
w1 = np.array([1.000,0.985944568893574,1.14987052049461])
def w_avg(input_values, w):
	weighted = np.average(input_values,weights=w)
	return weighted

def load_lines(table_name, country_one, country_two):
	country_string1 = f"{country_one}-{country_two}"
	country_string2 = f"{country_two}-{country_one}"
	query = f"SELECT geometry,user_id,dest_country,post_region,dest_region,region_move,cb_move,distance_km,row_id,country_code FROM {table_name} WHERE cb_move = '{country_string1}' OR cb_move = '{country_string2}' "
	lines_df = db_con.read_sql_inmem_uncompressed(query, db_engine)
	# Apply wkt
	lines_df['geometry'] = lines_df['geometry'].apply(wkt.loads)
	# Convert to GeoDataFrame
	lines_df_gdf = gpd.GeoDataFrame(lines_df, geometry='geometry')
	# CRS
	lines_df_gdf.crs = "EPSG:4326"
	# Delete dataframe
	del lines_df
	return lines_df_gdf['distance_km']

def country_pair_shapiro(country_one,country_two):
	lines_2017 = load_lines('nordic_2017_lines', country_one, country_two)
	lines_2018 = load_lines('nordic_2018_lines', country_one, country_two)
	lines_2019 = load_lines('pre_covid_lines', country_one, country_two)
	covid = load_lines('post_covid_lines', country_one, country_two)
	print(shapiro(lines_2017))
	print(shapiro(lines_2018))
	print(shapiro(lines_2019))
	print(shapiro(covid))
	return None

dk_fi = country_pair_shapiro('DK', 'FI')
dk_ic = country_pair_shapiro('DK', 'IS')
dk_no = country_pair_shapiro('DK', 'NO')
dk_sw = country_pair_shapiro('DK', 'SE')
fi_ic = country_pair_shapiro('FI', 'IS')
fi_no = country_pair_shapiro('FI', 'NO')
fi_sw = country_pair_shapiro('FI', 'SE')
ic_no = country_pair_shapiro('IS', 'NO')
ic_sw = country_pair_shapiro('IS', 'SE')
no_sw = country_pair_shapiro('NO', 'SE')

print('------------')

def mann_whitney_u_test(country_one,country_two):
	lines_2017 = load_lines('nordic_2017_lines', country_one, country_two)
	lines_2018 = load_lines('nordic_2018_lines', country_one, country_two)
	lines_2019 = load_lines('pre_covid_lines', country_one, country_two)
	covid = load_lines('post_covid_lines', country_one, country_two)
	alpha = 0.05
	stat, p = mannwhitneyu(lines_2017, covid)
	if p > alpha:
		print(f'2017 {country_one}-{country_two}')
		print('Same distribution (fail to reject H0)')
	else:
		print(f'2017 {country_one}-{country_two}')
		print('Different distribution (reject H0)')
	stat, p = mannwhitneyu(lines_2018, covid)
	if p > alpha:
		print(f'2018 {country_one}-{country_two}')
		print('Same distribution (fail to reject H0)')
	else:
		print(f'2018 {country_one}-{country_two}')
		print('Different distribution (reject H0)')
	stat, p = mannwhitneyu(lines_2019, covid)
	if p > alpha:
		print(f'2019 {country_one}-{country_two}')
		print('Same distribution (fail to reject H0)')
	else:
		print(f'2019 {country_one}-{country_two}')
		print('Different distribution (reject H0)')
	return None

dk_fi = mann_whitney_u_test('DK', 'FI')
dk_ic = mann_whitney_u_test('DK', 'IS')
dk_no = mann_whitney_u_test('DK', 'NO')
dk_sw = mann_whitney_u_test('DK', 'SE')
fi_ic = mann_whitney_u_test('FI', 'IS')
fi_no = mann_whitney_u_test('FI', 'NO')
fi_sw = mann_whitney_u_test('FI', 'SE')
ic_no = mann_whitney_u_test('IS', 'NO')
ic_sw = mann_whitney_u_test('IS', 'SE')
no_sw = mann_whitney_u_test('NO', 'SE')

print('------------')
print(shapiro(weighted_mean))
print(shapiro(mean_covid))

def kruskal_wallis(country_one,country_two):
	lines_2017 = load_lines('nordic_2017_lines', country_one, country_two)
	lines_2018 = load_lines('nordic_2018_lines', country_one, country_two)
	lines_2019 = load_lines('pre_covid_lines', country_one, country_two)
	covid = load_lines('post_covid_lines', country_one, country_two)
	alpha = 0.05
	stat, p = kruskal(lines_2017, lines_2018, lines_2019, covid)
	if p > alpha:
		print(f'{country_one}-{country_two}')
		print('Same distribution (fail to reject H0)')

	else:
		print(f'{country_one}-{country_two}')
		print('Different distribution (reject H0)')
		print(sp.posthoc_dunn([lines_2017, lines_2018, lines_2019, covid], p_adjust = 'bonferroni'))
	return None

dk_fi = kruskal_wallis('DK', 'FI')
dk_ic = kruskal_wallis('DK', 'IS')
dk_no = kruskal_wallis('DK', 'NO')
dk_sw = kruskal_wallis('DK', 'SE')
fi_ic = kruskal_wallis('FI', 'IS')
fi_no = kruskal_wallis('FI', 'NO')
fi_sw = kruskal_wallis('FI', 'SE')
ic_no = kruskal_wallis('IS', 'NO')
ic_sw = kruskal_wallis('IS', 'SE')
no_sw = kruskal_wallis('NO', 'SE')

def comb_mann_whitney_u_test(country_one,country_two):
	lines_2017 = load_lines('nordic_2017_lines', country_one, country_two)
	lines_2018 = load_lines('nordic_2018_lines', country_one, country_two)
	lines_2019 = load_lines('pre_covid_lines', country_one, country_two)
	covid = load_lines('post_covid_lines', country_one, country_two)
	comb = pd.concat([lines_2017,lines_2018, lines_2019])
	alpha = 0.001
	stat, p = mannwhitneyu(comb, covid)
	if p > alpha:
		print(f'{country_one}-{country_two}')
		print(f'Same distribution (fail to reject H0) - p-value: {round(p,3)}')
	else:
		print(f'{country_one}-{country_two}')
		print(f'Different distribution (reject H0) - p-value: {p}')
	return None

dk_fi = comb_mann_whitney_u_test('DK', 'FI')
dk_ic = comb_mann_whitney_u_test('DK', 'IS')
dk_no = comb_mann_whitney_u_test('DK', 'NO')
dk_sw = comb_mann_whitney_u_test('DK', 'SE')
fi_ic = comb_mann_whitney_u_test('FI', 'IS')
fi_no = comb_mann_whitney_u_test('FI', 'NO')
fi_sw = comb_mann_whitney_u_test('FI', 'SE')
ic_no = comb_mann_whitney_u_test('IS', 'NO')
ic_sw = comb_mann_whitney_u_test('IS', 'SE')
no_sw = comb_mann_whitney_u_test('NO', 'SE')

def load_all_lines(table_name):
	query = f"SELECT geometry,user_id,dest_country,post_region,dest_region,region_move,cb_move,distance_km,row_id,country_code FROM {table_name}"
	lines_df = db_con.read_sql_inmem_uncompressed(query, db_engine)
	# Apply wkt
	lines_df['geometry'] = lines_df['geometry'].apply(wkt.loads)
	# Convert to GeoDataFrame
	lines_df_gdf = gpd.GeoDataFrame(lines_df, geometry='geometry')
	# CRS
	lines_df_gdf.crs = "EPSG:4326"
	# Delete dataframe
	del lines_df
	return lines_df_gdf['distance_km']

def all_comb_mann_whitney_u_test():
	lines_2017 = load_all_lines('nordic_2017_lines')
	lines_2018 = load_all_lines('nordic_2018_lines')
	lines_2019 = load_all_lines('pre_covid_lines')
	covid = load_all_lines('post_covid_lines')
	comb = pd.concat([lines_2017,lines_2018, lines_2019])
	alpha = 0.05
	stat, p = mannwhitneyu(comb, covid)
	print('Overall')
	if p > alpha:
		print(f'Same distribution (fail to reject H0) - p-value: {round(p,3)}')
	else:
		print(f'Different distribution (reject H0) - p-value: {p}')
	return None
all_comb_mann_whitney_u_test()