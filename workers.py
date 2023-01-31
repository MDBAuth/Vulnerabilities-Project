
"""
    workers.py
    
    This is a python module that is saved as workers.py to the current working directory when the cell is run in the Jupyter environment
    
    workers.py is imported in the executable code below this cell for multiprocessing
    
    This convienently allows the python code of the workers.py module to be edited in the jupyter environment and stored with the multiprocessing code

"""
import os

import numpy as np
import pandas as pd
import fiona
from shapely import geometry


def shape_list(key, values, shapefile):
    """
        Get a generator of shapes from the given shapefile
            key: the key to match in 'properties' in the shape file
            values: a list of property values
            shapefile: the name of your shape file
            e.g. key='ORIGID', values=[1, 2, 3, 4, 5], 
            shapefile='/g/data/r78/DEA_Wetlands/shapefiles/MDB_ANAE_Aug2017_modified_2019_SB_3577.shp'
    """
    count = len(values)
    with fiona.open(shapefile) as allshapes:
        for shape in allshapes:
            shape_id = shape['properties'].get(key)
            if shape_id is None:
                continue
            if shape_id in values:
                yield(shape_id, shape)
                count -= 1
            if count <= 0:
                break
    
def get_areas(features, pkey='feature_id'):
    """
        Calculate the area of a list/generator of shapes
        input:
            features: a list of shapes indexed by the key
        output:
            a dataframe of area index by the key
    """
    re = pd.DataFrame()
    for f in features:
        va = pd.DataFrame([[f[0], geometry.shape(f[1]['geometry']).area/1e4]], columns=[pkey, 'area'])
        re = re.append(va, sort=False)
    return re.set_index(pkey)

    
def annual_metrics(wit_data, members=['pv', 'wet', 'water', 'bs', 'npv', ['npv', 'pv', 'wet'], ['pv', 'wet'], ['water', 'wet']], threshold=[25, 75], pkey='feature_id'):
                                              
    """
        Compute the annual max, min, mean, count with given wit data, members and threshold
        input:
            wit_data: dataframe of WIT
            members: the elements which the metrics are computed against, can be a column from wit_data, e.g. 'pv'
                         or the sum of wit columns, e.g. ['water', 'wet']
            threshold: a list of thresholds such that (elements >= threshold[i]) is True, 
                        where i = 0, 1...len(threshold)-1
        output:
            dataframe of metrics
    """
    years = wit_data['date']
    i = 0
    wit_df = wit_data.copy(deep=True)

    for m in members:
        if isinstance(m, list):
            wit_df.insert(wit_df.columns.size+i, '+'.join(m), wit_df[m].sum(axis=1))
    years = pd.DatetimeIndex(wit_df['date']).year.unique()
    shape_id_list = wit_df[pkey].unique()
    #shane changed 4 to 5 to accomodate median added below 
    wit_metrics = [pd.DataFrame()] * 5
    for y in years:
        wit_yearly = wit_df[pd.DatetimeIndex(wit_df['date']).year==y].drop(columns=['date']).groupby(pkey).max()
        wit_yearly.insert(0, 'year', y)
        wit_yearly = wit_yearly.rename(columns={n: n+'_max' for n in wit_yearly.columns[1:]})
        wit_metrics[0] = wit_metrics[0].append(wit_yearly, sort=False)
    for y in years:
        wit_yearly = wit_df[pd.DatetimeIndex(wit_df['date']).year==y].drop(columns=['date']).groupby(pkey).min()
        wit_yearly.insert(0, 'year', y)
        wit_yearly = wit_yearly.rename(columns={n: n+'_min' for n in wit_yearly.columns[1:]})
        wit_metrics[1] = wit_metrics[1].append(wit_yearly, sort=False)
    for y in years:
        wit_yearly = wit_df[pd.DatetimeIndex(wit_df['date']).year==y].drop(columns=['date']).groupby(pkey).mean()
        wit_yearly.insert(0, 'year', y)
        wit_yearly = wit_yearly.rename(columns={n: n+'_mean' for n in wit_yearly.columns[1:]})
        wit_metrics[2] = wit_metrics[2].append(wit_yearly, sort=False)
        
    #*********************** START ADDED BY SHANE ***********************
    #adding median
    for y in years:
        wit_yearly = wit_df[pd.DatetimeIndex(wit_df['date']).year==y].drop(columns=['date']).groupby(pkey).median()
        wit_yearly.insert(0, 'year', y)
        wit_yearly = wit_yearly.rename(columns={n: n+'_median' for n in wit_yearly.columns[1:]})
        wit_metrics[3] = wit_metrics[3].append(wit_yearly, sort=False)
    #*********************** END ADDED BY SHANE ***********************      
    for y in years:
        wit_yearly = wit_df[pd.DatetimeIndex(wit_df['date']).year==y][[pkey, 'bs']].groupby(pkey).count()
        wit_yearly.insert(0, 'year', y)
        wit_yearly = wit_yearly.rename(columns={n: 'count' for n in wit_yearly.columns[1:]})
        #shane changed index from 3 to 4 to accomodate median added above 
        wit_metrics[4] = wit_metrics[4].append(wit_yearly, sort=False)
    #for t in threshold:
    #    wit_df_ts = wit_df.copy(deep=True)
    #    wit_metrics += [pd.DataFrame()]
    #    wit_df_ts.loc[:, wit_df_ts.columns[2:]] = wit_df_ts.loc[:, wit_df_ts.columns[2:]].mask((wit_df_ts[wit_df_ts.columns[2:]] < t/100), np.nan)
    #    for y in years:
    #        wit_yearly = wit_df_ts[pd.DatetimeIndex(wit_df_ts['date']).year==y].drop(columns=['date']).groupby(pkey).count()
    #        wit_yearly.insert(0, 'year', y)
    #        wit_yearly = wit_yearly.rename(columns={n: n+'_count'+str(t) for n in wit_yearly.columns[1:]})
    #        wit_metrics[-1] = wit_metrics[-1].append(wit_yearly, sort=False)
    wit_yearly_metrics = wit_metrics[0]
    wit_yearly_metrics.sort_values(by=[pkey, 'year'],inplace=True)
    for i in range(len(wit_metrics)-1):
        wit_yearly_metrics = pd.merge(wit_yearly_metrics, wit_metrics[i+1], on=[pkey, 'year'], how='inner')
    ofn = "WIT_ANAE_yearly_metrics"+str(wit_data['chunk'].iat[0])+".csv"
    wit_yearly_metrics.to_csv(ofn)
    return wit_yearly_metrics

def get_event_time(wit_ww, threshold, pkey='feature_id'):
    """

        Compute inundation event time by given threshold
        input:
            wit_df: wetness computed from wit data
            threshold: a value such that (water+wet > threshold) = inundation
        output:
            dateframe of inundation event time
    """
    if isinstance(threshold, pd.DataFrame):
        gid = wit_ww.index.unique()[0]
        poly_threshold = threshold.loc[gid].to_numpy()[0]
    else:
        poly_threshold = threshold
    i_start = wit_ww[wit_ww['water+wet'] >= poly_threshold]['date'].min()
    if pd.isnull(i_start):
        re = pd.DataFrame([[np.nan] * 5], columns=['threshold', 'start_time', 'end_time', 'duration', 'gap'], index=wit_ww.index.unique())
        re.index.name = pkey
        return re
    #SSB - moved equal to needed for when threshold = 0
    #re_idx = np.searchsorted(wit_ww[(wit_ww['water+wet'] < poly_threshold)]['date'].values, 
    #                         wit_ww[(wit_ww['water+wet'] >= poly_threshold)]['date'].values)
    re_idx = np.searchsorted(wit_ww[(wit_ww['water+wet'] <= poly_threshold)]['date'].values,
                         wit_ww[(wit_ww['water+wet'] > poly_threshold)]['date'].values)

    re_idx, count = np.unique(re_idx, return_counts=True)
    start_idx = np.zeros(len(count)+1, dtype='int')
    start_idx[1:] = np.cumsum(count)

    #SSB removed "equals" sorts correctly when threshold is zero
    #re_start = wit_ww[(wit_ww['water+wet'] >= poly_threshold)].iloc[start_idx[:-1]][['date']].rename(columns={'date': 'start_time'})
    #re_end = wit_ww[(wit_ww['water+wet'] >= poly_threshold)].iloc[start_idx[1:] - 1][['date']].rename(columns={'date': 'end_time'})
    re_start = wit_ww[(wit_ww['water+wet'] > poly_threshold)].iloc[start_idx[:-1]][['date']].rename(columns={'date': 'start_time'})
    re_end = wit_ww[(wit_ww['water+wet'] > poly_threshold)].iloc[start_idx[1:] - 1][['date']].rename(columns={'date': 'end_time'})

    re = pd.concat([re_start, re_end], axis=1)
    if not re.empty:
        re.insert(2, 'duration', 
                  (re['end_time'] - re['start_time'] + np.timedelta64(1, 'D')).astype('timedelta64[D]').astype('timedelta64[D]'))
        re.insert(3, 'gap', np.concatenate([[np.timedelta64(0, 'D')],
                                            (re['start_time'][1:].values - re['end_time'][:-1].values - np.timedelta64(1, 'D')).astype('timedelta64[D]')]))
        re.insert(0, 'threshold', poly_threshold)
        re.insert(0, pkey, wit_ww.index.unique()[0])
        re = re.set_index(pkey)
    return re
    
def get_im_stats(grouped_wit, im_time, wit_area):
    """
        Get inundation stats given wit data and events
        input:
            grouped_wit: wit data
            im_time: inundation events in time
        output:
            the stats of inundation events
    """
    gid = grouped_wit.index.unique()[0]
    if gid not in im_time.indices.keys():
        return pd.DataFrame([[np.nan]*5], columns=['start_time', 'max_water+wet', 'mean_water+wet', 'max_wet_area', 'mean_wet_area'],
                           index=[gid])
    re_left = np.searchsorted(grouped_wit['date'].values.astype('datetime64'),
                         im_time.get_group(gid)['start_time'].values, side='left')
    re_right = np.searchsorted(grouped_wit['date'].values.astype('datetime64'),
                         im_time.get_group(gid)['end_time'].values, side='right')
    re = pd.DataFrame()
    for a, b in zip(re_left, re_right):
        tmp = pd.concat([grouped_wit.iloc[a:a+1]['date'].rename('start_time').astype('datetime64'),
                         pd.Series(grouped_wit.iloc[a:b]['water+wet'].max(),index=[gid], name='max_water+wet'),
                         pd.Series(grouped_wit.iloc[a:b]['water+wet'].mean(),index=[gid], name='mean_water+wet')],
                        axis=1)
        if isinstance(wit_area, pd.DataFrame):
            tmp.insert(3, 'max_wet_area', tmp['max_water+wet'].values * wit_area[wit_area.index==gid].values)
            tmp.insert(4, 'mean_wet_area', tmp['mean_water+wet'].values * wit_area[wit_area.index==gid].values)
        re = re.append(tmp, sort=False)
    #reset the index as the pkey
    re.index.name = grouped_wit.index.name
    re.reset_index()
    return re

def event_time(wit_df, threshold=0.01, pkey='feature_id'):
    """
        Compute the inundation events with given wit data and threshold
        input:
            wit_df: wetness computed from wit data
            threshold: a value such that (water+wet > threshold) = inundation,
        output:
            dataframe of events
    """
    return wit_df.groupby(pkey).apply(get_event_time, threshold=threshold, pkey=pkey).dropna().droplevel(0)

def event_stats(wit_df, wit_im, wit_area, pkey='feature_id'):
    """
        Compute inundation event stats with given wit wetness, events defined by (start_time, end_time) 
        and polygon areas
        input:
            wit_df: wetness computed from wit data
            wit_im: inundation event
            wit_area: polygon areas indexed by the key
        output:
            dataframe of event stats
    """
    grouped_im = wit_im[['start_time', 'end_time']].groupby(pkey)
    #was droplevel(0) but this left first column without a header (1) deletes that column instead
    return wit_df.groupby(pkey).apply(get_im_stats, im_time=grouped_im, wit_area=wit_area).droplevel(1)

def inundation_metrics(wit_data, threshold=0.01, shapefile = 'shapefile', skey='UID', pkey='feature_id'):
    """
        Compute inundation metrics with given wit data, polygon areas and threshold
        input:
            wit_data: a dataframe of wit_data
            wit_area: polygon areas indexed by the key
            threshold: a value such that (water+wet > threshold) = inundation
        output:
            dataframe of inundation metrics
    """
    wit_area=[]
    if os.path.isfile(shapefile):
        features = shape_list(skey, wit_data['feature_id'].unique(), shapefile)
        wit_area = get_areas(features, pkey='feature_id')

    wit_df = wit_data.copy(deep=True)
    wit_df.insert(2, 'water+wet', wit_df[['water', 'wet']].sum(axis=1).round(decimals = 4))
    #wit_df = wit_df.drop(columns=wit_df.columns[3:])
    wit_df = wit_df[[pkey,'date','water+wet']]
    wit_df['date'] = wit_df['date'].astype('datetime64')
    wit_df = wit_df.set_index(pkey)
    wit_im_time = event_time(wit_df, threshold, pkey='feature_id')
    ofn = "WIT_ANAE_event_times"+str(wit_data['chunk'].iat[0])+".csv"
    wit_im_time.to_csv(ofn)
    
    wit_im_stats = event_stats(wit_df, wit_im_time, wit_area, pkey='feature_id')
    ofn = "WIT_ANAE_event_stats"+str(wit_data['chunk'].iat[0])+".csv"
    wit_im_stats.to_csv(ofn)
    
    wit_im = pd.DataFrame()
    if not wit_im_time.empty:
        wit_im =pd.merge(wit_im_time, wit_im_stats, on=[pkey, 'start_time'], how='inner')
        ofn = "WIT_ANAE_inundation_metrics"+str(wit_data['chunk'].iat[0])+".csv"
        wit_im.to_csv(ofn)
    return wit_im

def interpolate_daily(wit_data, pkey='feature_id'):
    return wit_data.groupby(pkey).apply(interpolate_wit, pkey=pkey).droplevel(0)

def interpolate_wit(grouped_wit, pkey='feature_id'):
    daily_wit = pd.DataFrame({pkey: grouped_wit[pkey].unique()[0], 'date': pd.date_range(grouped_wit['date'].astype('datetime64[D]').min(), grouped_wit['date'].astype('datetime64[D]').max(), freq='D'),
                          'bs': np.nan, 'npv': np.nan, 'pv': np.nan, 'wet': np.nan, 'water': np.nan})
    _, nidx, oidx = np.intersect1d(daily_wit['date'].to_numpy().astype('datetime64[D]'), grouped_wit['date'].to_numpy().astype('datetime64[D]'),
                  return_indices=True)
    daily_wit.loc[nidx, ["bs","npv","pv","wet","water"]]  = grouped_wit[["bs","npv","pv","wet","water"]].iloc[oidx].to_numpy()
    #daily_wit = daily_wit.interpolate(axis=0)
    #recent version of pandas throws error due to date column.  workaround is to only interpolate the columns of data
    daily_wit[["bs","npv","pv","wet","water"]] = daily_wit.groupby(['feature_id']).apply(lambda x: x[["bs","npv","pv","wet","water"]].interpolate(axis=0))
    if 'chunk' in grouped_wit.columns:
        daily_wit['chunk'] = grouped_wit['chunk'].unique()[0]
    return daily_wit

def time_since_last_inundation(wit_data, wit_im, pkey='feature_id'):
    """
        create a pivot table to gather the time since last inundation using the event metrics
        timesincelast = number of days from last event end-date to final date in WIT record
    """
    maxdate = pd.pivot_table(wit_data, index=pkey, values=['date'], aggfunc=np.max)
    maxdate['final-date'] = maxdate['date'].astype('datetime64')
    lastevent = pd.pivot_table(wit_im, index=pkey, values=['end_time'], aggfunc=np.max)
    time_since_last = pd.merge(maxdate, lastevent, on=[pkey], how='inner')
    time_since_last.insert(2, 'timesincelast', 
              (time_since_last['final-date'] - time_since_last['end_time']).astype('timedelta64[D]'))
    ofn = "WIT_ANAE_time_since_last_inundation"+str(wit_data['chunk'].iat[0])+".csv"
    time_since_last.to_csv(ofn)
    return time_since_last

def all_time_median(wit_data, members=[['water', 'wet']], pkey='feature_id'):
    """
        Compute the all time median
        input:
            wit_data: dataframe of WIT
            members: the elements which the metrics are computed against, can be a column from wit_data, e.g. 'pv'
                         or the sum of wit columns, e.g. ['water', 'wet']
        output:
            dataframe of median indexed by pkey
    """
    wit_df = wit_data.copy(deep=True)
    i = 0
    for m in members:
        if isinstance(m, list):
            wit_df.insert(wit_df.columns.size+i, '+'.join(m), wit_df[m].sum(axis=1))
        i += 1
    wit_median = wit_df.groupby(pkey).median().round(decimals = 4)
    ofn = "WIT_ANAE_event_threshold"+str(wit_data['chunk'].iat[0])+".csv"
    wit_median.to_csv(ofn)
    return wit_median
