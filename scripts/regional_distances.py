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
from datetime import datetime
import db_connection as db_con

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

def regional_distance_stats(table_name, region_one,region_two):
	regions_string = f"{region_one}-{region_two}|{region_two}-{region_one}"
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
	
	subset = lines_df_gdf[lines_df_gdf['region_move'].str.match(regions_string)]
	absolute_numbers = np.array([len(subset)])

	# Distances
	median_dist = np.array([subset['distance_km'].median()])
	mean_dist = np.array([subset['distance_km'].mean()])
	std_dist = np.array([subset['distance_km'].std()])
	return median_dist, mean_dist, std_dist

def regional_weighted_combination(region_one, region_two, w):
	median_2017, mean_2017, std_2017 = regional_distance_stats("nordic_2017_lines", region_one, region_two)
	median_2018, mean_2018, std_2018 = regional_distance_stats("nordic_2018_lines", region_one, region_two)
	median_2019, mean_2019, std_2019 = regional_distance_stats("pre_covid_lines", region_one, region_two)
	meds = np.array([median_2017[0],median_2018[0],median_2019[0]])
	means = np.array([mean_2017[0],mean_2018[0],mean_2019[0]])
	stds = np.array([std_2017[0],std_2018[0],std_2019[0]])
	weighted_med = w_avg(meds,w1)
	weighted_mean = w_avg(means,w1)
	weighted_std = w_avg(stds,w1)

	return weighted_med, weighted_mean, weighted_std

print(regional_weighted_combination('SE33','FI1D', w1))

def multi_regional_distance_stats(table_name, regions_country_one,regions_country_two):
	region_pairs = []
	for region in regions_country_one:
		for reg in range(len(regions_country_two)):
			appendstring = f"{region}-{regions_country_two[reg]}"
			reverse_append = f"{regions_country_two[reg]}-{region}"
			region_pairs.append(appendstring)
			region_pairs.append(reverse_append)

	regions_string = '|'.join(region_pairs)
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

	subset = lines_df_gdf[lines_df_gdf['region_move'].str.match(regions_string)]
	absolute_numbers = np.array([len(subset)])

	# Distances
	median_dist = np.array([subset['distance_km'].median()])
	mean_dist = np.array([subset['distance_km'].mean()])
	std_dist = np.array([subset['distance_km'].std()])
	return median_dist, mean_dist, std_dist

def multi_regional_weighted_combination(regions_country_one, regions_country_two, w):
	median_2017, mean_2017, std_2017 = multi_regional_distance_stats("nordic_2017_lines", regions_country_one, regions_country_two)
	median_2018, mean_2018, std_2018 = multi_regional_distance_stats("nordic_2018_lines", regions_country_one, regions_country_two)
	median_2019, mean_2019, std_2019 = multi_regional_distance_stats("pre_covid_lines", regions_country_one, regions_country_two)
	meds = np.array([median_2017[0],median_2018[0],median_2019[0]])
	
	means = np.array([mean_2017[0],mean_2018[0],mean_2019[0]])
	stds = np.array([std_2017[0],std_2018[0],std_2019[0]])
	weighted_med = w_avg(meds,w1)
	weighted_mean = w_avg(means,w1)
	weighted_std = w_avg(stds,w1)

	return weighted_med, weighted_mean, weighted_std
print(multi_regional_weighted_combination(['DK01','DK02','DK03','DK04','DK05'],['SE22','SE23','SE21'], w1))
print(multi_regional_distance_stats('post_covid_lines',['DK01','DK02','DK03','DK04','DK05'],['SE22','SE23','SE21']))