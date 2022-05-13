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
from shapely.geometry import Point, LineString, Polygon
import geojson
import folium
import contextily as ctx
from shapely import wkt
import sys
from scipy.stats import shapiro
from datetime import datetime
import db_connection as db_con
starttime = datetime.now()

# Import credential and connection settings
db_name = db_con.db_name
db_username = db_con.db_username
db_host = db_con.db_host
db_port = db_con.db_port
db_pwd = db_con.db_pwd
engine_string = f"postgresql://{db_username}:{db_pwd}@{db_host}:{db_port}/{db_name}"
db_engine = create_engine(engine_string)

def get_median_distances(table_name):
	print(table_name)
	query = f'SELECT geometry,user_id,dest_country,orig_time,dest_time,duration,post_region,dest_region,region_move,cb_move,distance_km,row_id,country_code FROM {table_name}'
	lines_df = db_con.read_sql_inmem_uncompressed(query, db_engine)
	# Apply wkt
	lines_df['geometry'] = lines_df['geometry'].apply(wkt.loads)
	# Convert to GeoDataFrame
	lines_df_gdf = gpd.GeoDataFrame(lines_df, geometry='geometry')
	# CRS
	lines_df_gdf.crs = "EPSG:4326"
	# Delete dataframe
	del lines_df
	# Convert timestamps
	lines_df_gdf['orig_time'] = lines_df_gdf['orig_time'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d-%H"))
	lines_df_gdf['dest_time'] = lines_df_gdf['dest_time'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d-%H"))
	# Add month
	lines_df_gdf['month'] = lines_df_gdf['dest_time'].apply(lambda x: x.month)
	lines_df_gdf['yearmonth'] = lines_df_gdf['dest_time'].apply(lambda x: int(str(x.year)+str(x.month).zfill(2)))
	# Add day of week
	lines_df_gdf['orig_day_of_week'] = lines_df_gdf['orig_time'].apply(lambda x: x.dayofweek)
	lines_df_gdf['dest_day_of_week'] = lines_df_gdf['dest_time'].apply(lambda x: x.dayofweek)

	dk_fi = lines_df_gdf[lines_df_gdf['cb_move'].str.match('DK-FI|FI-DK')]
	dk_ic = lines_df_gdf[lines_df_gdf['cb_move'].str.match('DK-IS|IS-DK')]
	dk_no = lines_df_gdf[lines_df_gdf['cb_move'].str.match('DK-NO|NO-DK')]
	dk_sw = lines_df_gdf[lines_df_gdf['cb_move'].str.match('DK-SE|SE-DK')]
	fi_ic = lines_df_gdf[lines_df_gdf['cb_move'].str.match('FI-IS|IS-FI')]
	fi_no = lines_df_gdf[lines_df_gdf['cb_move'].str.match('FI-NO|NO-FI')]
	fi_sw = lines_df_gdf[lines_df_gdf['cb_move'].str.match('FI-SE|SE-FI')]
	ic_no = lines_df_gdf[lines_df_gdf['cb_move'].str.match('IS-NO|NO-IS')]
	ic_sw = lines_df_gdf[lines_df_gdf['cb_move'].str.match('IS-SE|SE-IS')]
	no_sw = lines_df_gdf[lines_df_gdf['cb_move'].str.match('NO-SE|SE-NO')]
	absolute_numbers = np.array([len(dk_fi), len(dk_ic), len(dk_no), len(dk_sw), len(fi_ic), len(fi_no), len(fi_sw), len(ic_no), len(ic_sw), len(no_sw)])

	# Distances
	median_dist = np.array([dk_fi['distance_km'].median(), dk_ic['distance_km'].median(), dk_no['distance_km'].median(), dk_sw['distance_km'].median(), fi_ic['distance_km'].median(), fi_no['distance_km'].median(), fi_sw['distance_km'].median(), ic_no['distance_km'].median(), ic_sw['distance_km'].median(), no_sw['distance_km'].median()])
	return absolute_numbers, median_dist

w1 = np.array([1.000,0.985944568893574,1.14987052049461])
def w_avg(input_values, w):
	weighted = np.average(input_values,weights=w)
	return weighted

ab_2017, med_2017 = get_median_distances("nordic_2017_lines")
ab_2018, med_2018 = get_median_distances("nordic_2018_lines")
ab_2019, med_2019 = get_median_distances("pre_covid_lines")
ab_covid, med_covid = get_median_distances("post_covid_lines")


all_abs = []
for i in range(10):
	ab_vals = np.array([ab_2017[i], ab_2018[i], ab_2019[i]])
	all_abs.append(int(round(w_avg(ab_vals, w1),0)))
	med_vals = np.array([med_2017[i], med_2018[i], med_2019[i]])

print(f"[INFO] -- All abosulte")
print(all_abs)
print(sum(all_abs))

overall_median = np.array([485,472,435])
overall_absolute = np.array([59208,53031,44126])
print(w_avg(overall_absolute,w1))

# Weighted Movement shares
shares = []
for i in all_abs:
	shares.append(round((i/w_avg(overall_absolute,w1))*100,3))
print(f"Shares weighted average: {shares}")

# Weighted absolute change
all_abs.append(int(round(w_avg(overall_absolute,w1),0)))

ab_covid = list(ab_covid)
ab_covid.append(sum(ab_covid))
print(ab_covid)
for i in range(len(ab_covid)):
	print(ab_covid[i]-all_abs[i])

def distance_stats(table_name):
	print(table_name)
	query = f'SELECT geometry,user_id,dest_country,orig_time,dest_time,duration,post_region,dest_region,region_move,cb_move,distance_km,row_id,country_code FROM {table_name}'
	lines_df = db_con.read_sql_inmem_uncompressed(query, db_engine)
	# Apply wkt
	lines_df['geometry'] = lines_df['geometry'].apply(wkt.loads)
	# Convert to GeoDataFrame
	lines_df_gdf = gpd.GeoDataFrame(lines_df, geometry='geometry')
	# CRS
	lines_df_gdf.crs = "EPSG:4326"
	# Delete dataframe
	del lines_df
	# Convert timestamps
	lines_df_gdf['orig_time'] = lines_df_gdf['orig_time'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d-%H"))
	lines_df_gdf['dest_time'] = lines_df_gdf['dest_time'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d-%H"))
	# Add month
	lines_df_gdf['month'] = lines_df_gdf['dest_time'].apply(lambda x: x.month)
	lines_df_gdf['yearmonth'] = lines_df_gdf['dest_time'].apply(lambda x: int(str(x.year)+str(x.month).zfill(2)))
	# Add day of week
	lines_df_gdf['orig_day_of_week'] = lines_df_gdf['orig_time'].apply(lambda x: x.dayofweek)
	lines_df_gdf['dest_day_of_week'] = lines_df_gdf['dest_time'].apply(lambda x: x.dayofweek)
	dk_fi = lines_df_gdf[lines_df_gdf['cb_move'].str.match('DK-FI|FI-DK')]
	dk_ic = lines_df_gdf[lines_df_gdf['cb_move'].str.match('DK-IS|IS-DK')]
	dk_no = lines_df_gdf[lines_df_gdf['cb_move'].str.match('DK-NO|NO-DK')]
	dk_sw = lines_df_gdf[lines_df_gdf['cb_move'].str.match('DK-SE|SE-DK')]
	fi_ic = lines_df_gdf[lines_df_gdf['cb_move'].str.match('FI-IS|IS-FI')]
	fi_no = lines_df_gdf[lines_df_gdf['cb_move'].str.match('FI-NO|NO-FI')]
	fi_sw = lines_df_gdf[lines_df_gdf['cb_move'].str.match('FI-SE|SE-FI')]
	ic_no = lines_df_gdf[lines_df_gdf['cb_move'].str.match('IS-NO|NO-IS')]
	ic_sw = lines_df_gdf[lines_df_gdf['cb_move'].str.match('IS-SE|SE-IS')]
	no_sw = lines_df_gdf[lines_df_gdf['cb_move'].str.match('NO-SE|SE-NO')]
	absolute_numbers = np.array([len(dk_fi), len(dk_ic), len(dk_no), len(dk_sw), len(fi_ic), len(fi_no), len(fi_sw), len(ic_no), len(ic_sw), len(no_sw)])

	# Distances
	median_dist = np.array([dk_fi['distance_km'].median(), dk_ic['distance_km'].median(), dk_no['distance_km'].median(), dk_sw['distance_km'].median(), fi_ic['distance_km'].median(), fi_no['distance_km'].median(), fi_sw['distance_km'].median(), ic_no['distance_km'].median(), ic_sw['distance_km'].median(), no_sw['distance_km'].median()])
	mean_dist = np.array([dk_fi['distance_km'].mean(), dk_ic['distance_km'].mean(), dk_no['distance_km'].mean(), dk_sw['distance_km'].mean(), fi_ic['distance_km'].mean(), fi_no['distance_km'].mean(), fi_sw['distance_km'].mean(), ic_no['distance_km'].mean(), ic_sw['distance_km'].mean(), no_sw['distance_km'].mean()])
	std_dist = np.array([dk_fi['distance_km'].std(), dk_ic['distance_km'].std(), dk_no['distance_km'].std(), dk_sw['distance_km'].std(), fi_ic['distance_km'].std(), fi_no['distance_km'].std(), fi_sw['distance_km'].std(), ic_no['distance_km'].std(), ic_sw['distance_km'].std(), no_sw['distance_km'].std()])
	total_median = int(round(lines_df_gdf['distance_km'].median(),2))
	total_mean = int(round(lines_df_gdf['distance_km'].mean(),2))
	total_std = int(round(lines_df_gdf['distance_km'].std(),2))
	return median_dist, mean_dist, std_dist, total_median, total_mean, total_std

def weighted_combination(input_1,input_2,input_3, w):
	days_bl1 = np.array(input_1)
	days_bl2 = np.array(input_2)
	days_bl3 = np.array(input_3)
	weighted = []
	for i in range(len(days_bl1)):
		day_vals = np.array([days_bl1[i], days_bl2[i], days_bl3[i]])
		weighted.append(int(round( w_avg(day_vals,w1),2 )))

	return weighted
print("------------")

median_2017, mean_2017, std_2017, t_median_2017 ,t_mean_2017 , t_std_2017 = distance_stats("nordic_2017_lines")
median_2018, mean_2018, std_2018, t_median_2018 ,t_mean_2018 , t_std_2018 = distance_stats("nordic_2018_lines")
median_2019, mean_2019, std_2019, t_median_2019 ,t_mean_2019 , t_std_2019 = distance_stats("pre_covid_lines")
median_covid , mean_covid , std_covid, t_median_covid ,t_mean_covid , t_std_covid = distance_stats("post_covid_lines")
print(f"[INFO] -- Baseline")
print("Mean")
weighted_mean = weighted_combination(mean_2017, mean_2018, mean_2019, w1)
print(weighted_combination(mean_2017, mean_2018, mean_2019, w1))
print(w_avg(np.array([t_mean_2017,t_mean_2018,t_mean_2019]), w1))
print("Median")
print(weighted_combination(median_2017, median_2018, median_2019, w1))
print(w_avg(np.array([t_median_2017,t_median_2018,t_median_2019]), w1))

print("Std")
print(weighted_combination(std_2017, std_2018, std_2019, w1))
print(w_avg(np.array([t_std_2017,t_std_2018,t_std_2019]), w1))

print(f"[INFO] -- COVID")
print("Mean")
mean_covid = [int(round(mean,0)) for mean in mean_covid]
print(mean_covid)
print(t_mean_covid)
print("Median")
median_covid = [int(round(median,0)) for median in median_covid]
print(median_covid)
print(t_median_covid)

print("Std")
std_covid = [int(round(std,0)) for std in std_covid]
print(std_covid)
print(t_std_covid)

print(f'Script took: {datetime.now()-starttime}')