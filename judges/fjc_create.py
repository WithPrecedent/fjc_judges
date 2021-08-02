"""
fjc_create: functions for acquiring and formatting FJC judge data
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2020-2021, Corey Rayburn Yung
License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import annotations
import csv
import pathlib
import re
from shutil import copyfile
import warnings

import numpy as np
import pandas as pd
import requests

from judges import instructions


# def download_file(url: str, 
#                   file_path: pathlib.Path, 
#                   session: requests.Session = None):
#     """Downloads file at 'url' to 'file_path'."""
#     tool = session or requests
#     response = getattr(tool, 'get')(url)
#     with open(file_path, 'wb') as downloaded:  
#         downloaded.write(response.content)
#     return

# def download_fjc():
#     """Downloads all fjc files based on module constants."""
#     session = requests.Session()
#     for source in ['SERVICE', 'CAREER', 'DEMOGRAPHICS']:
#         str_path = getattr(instructions, f'{source}_PATH']
#         file_path = pathlib.Path(str_path)
#         if not file_path.is_file():
#             url = getattr(instructions, f'{source}_URL']
#             download_file(url = url, file_path = file_path, session = session)
#     return

def load_file(file_path: pathlib.Path, columns: list[str] = None):
    """Loads file as a pandas DataFrame"""
    kwargs = {'index_col': False, 'encoding': 'windows-1252'}
    if columns:
        kwargs['usecols'] = columns
    return pd.read_csv(file_path, **kwargs)

def load_fjc() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads and returns all FJC judge files into pandas DataFrames."""
    dfs = []
    for source in ['SERVICE', 'CAREER', 'DEMOGRAPHICS']:
        str_path = getattr(instructions, f'{source}_PATH')
        file_path = pathlib.Path(str_path)
        df = load_file(file_path = file_path)
        dfs.append(df)
    return dfs

def combine_fjc(service: pd.DataFrame, 
                career: pd.DataFrame, 
                demographics: pd.DataFrame) -> pd.DataFrame:
    """Combines fjc DataFrames into a single pandas DataFrame."""
    combined = pd.merge(service, 
                        career, 
                        on = 'nid', 
                        suffixes = {'', '_EXTRA'})
    combined = pd.merge(combined, 
                        demographics, 
                        on = 'nid', 
                        suffixes = {'', '_EXTRA'})
    return combined
            
def fix_columns(df: pd.DataFrame) -> pd.DataFrame:  
    """Makes basic adjustments to column names and replaces empty data."""
    df = (df.rename(columns = str.lower)
        .rename(columns = lambda x: x.replace(' ', '_'))
        .rename(columns = instructions.SERVICE_RENAMES)
        .rename(columns = instructions.CAREER_RENAMES)
        .rename(columns = instructions.DEMOGRAPHICS_RENAMES)
        .replace(['None (assignment)', 'None (reassignment)'], 
                 method = 'ffill')
        .pipe(fill_empty)
        .astype(dtype = {'nid' : str}))
    return df     
         
def munge_fjc(df: pd.DataFrame) -> pd.DataFrame:
    """Adds needed data to fjc judges DataFrame."""
    df = fix_columns(df = df)
    df = df[instructions.ALL_COLUMNS]
    df['court_num'] = df['court'].map(instructions.COURT_NUMBERS)
    df['circuit_num'] = df['court'].map(instructions.CIRCUIT_NUMBERS)
    df = (df.astype(dtype = {'nid' : int})
            .astype(dtype = {'senate_vote' : str})
            .pipe(encode_bio_data)
            .apply(encode_senate_vote, axis = 'columns')
            .fillna('')
            .pipe(time_limit)
            # .set_index('nid')
            .pipe(name_changes)
            .apply(name_perms, axis = 'columns')
            .reset_index())
    return df       

def fill_empty(df: pd.DataFrame) -> pd.DataFrame:
    df['senate_vote_type'].replace(np.nan, method = 'ffill', inplace = True)
    df['senate_vote'].replace(np.nan, method = 'ffill', inplace = True)
    return df

def encode_bio_data(df: pd.DataFrame) -> pd.DataFrame:
    df['recess'] = np.where(df['recess'].str.len() > 1, True, False)
    df['party'] = np.where(df['party'] == 'Democratic', -1, 1)
    df['aba_rating'] = (np.where(df['aba_rating'] 
                        == 'Exceptionally Well Qualified', 4,
                        np.where(df['aba_rating'] 
                                    == 'Well Qualified', 3,
                        np.where(df['aba_rating'] 
                                    == 'Qualified', 2,
                        np.where(df['aba_rating'] 
                                    == 'Not Qualified', 1, 0)))))
    df['woman'] = np.where(df['gender'] == 'Female', True, False)
    df['minority'] = (np.where(df['race'].str.contains('American')
                                | df['race'].str.contains('Hispanic')
                                | df['race'].str.contains('Pacific'), True, 
                                False))
    df['pres_num'] = df['president'].map(instructions.PRESIDENTS)
    return df

def encode_senate_vote(row: pd.Series) -> pd.Series:
    if row['senate_vote_type'] == 'Voice':
        row['senate_percent'] = 1
    elif '/' in row['senate_vote'] and not '//' in row['senate_vote']:
        yeas, neas = row['senate_vote'].split('/')
        row['senate_percent'] = int(yeas)/(int(yeas) + int(neas)) 
    else:
        row['senate_percent'] = 0
    return row

def time_limit(df: pd.DataFrame) -> pd.DataFrame:
    """Limits data to time period when judge was on the bench."""
    df['termination_date'] = pd.to_datetime(df['termination_date'])
    df['end_year'] = df['termination_date'].dt.year
    df['end_year'] = df['end_year'].replace(np.nan, instructions.END_YEAR)
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['start_year'] = df['start_date'].dt.year
    df = df[df['end_year'] > instructions.START_YEAR - 2]
    df = df[df['start_year'] <= instructions.END_YEAR] 
    return df
    
def name_changes(df: pd.DataFrame) -> pd.DataFrame:
    """Adds rows for known omissions or errors in the FJC data."""  
    def make_change(df, index_num, col_name, value):
        row = df.loc[index_num]
        row[col_name] = value
        return row   
    try:
        df = df.append(make_change(df, 1386716, 'last_name', 'Randall'))
    except KeyError:
        pass
    try:
        df = df.append(make_change(df, 1382851, 'first_name', 'Sam')) 
    except KeyError:
        pass 
    return df
        
def name_perms(row: pd.Series) -> pd.Series:
    """Constructs a consistent set of name permutations for each judge."""
    row['first_name'] = re.sub("\.|\,|\[|\]|\'", '', row['first_name'])
    row['last_name'] = re.sub("\.|\,|\[|\]|\'", '', row['last_name'])
    row['middle_name'] = re.sub("\.|\,|\[|\]|\'", '', row['middle_name'])
    first = row['first_name'].strip().upper()
    first_init = first[0].upper()
    middle = row['middle_name'].strip().upper()
    if middle:    
        middle_init = middle[0].upper()
    else:
        middle_init = ''
    last = row['last_name'].strip().upper()
    row['name_perm1'] = f'{first} {middle} {last}'
    row['name_perm2'] = f'{first} {middle_init} {last}'
    row['name_perm3'] = f'{first} {last}'
    row['name_perm4'] = f'{first_init} {middle} {last}'
    row['name_perm5'] = f'{first_init} {middle_init} {last}'
    row['name_perm6'] = f'{first_init} {last}'
    row['name_perm7'] = f'{last}'
    return row

def make_names_df(df: pd.DataFrame) -> pd.DataFrame:  
    df = df[['start_year', 'end_year', 'court_num', 'circuit_num', 
             'name_perm1', 'name_perm2', 'name_perm3', 'name_perm4', 
             'name_perm5', 'name_perm6', 'name_perm7', 'judge_name']]
    df.sort_values(by = ['court_num'])
    df['year'] = 0
    df['nindex'] = range(0, len(df))
    df = pd.wide_to_long(df, 
                         stubnames = 'name_perm', 
                         i = 'nindex', 
                         j = 'nperm')        
    df = df[df.name_perm != '']
    df.drop_duplicates(ignore_index = True, inplace = True)
    return df

def convert_judge_name(year, court, name, dict_type):
    if dict_type == 1:
        return str(int(court) * 10000 + int(year)) + name
    elif dict_type == 2:
        return str(int(year)) + name
    else:
        return name
        
def convert_judge_name_series(year, court, name, dict_type):
    if dict_type == 1:
        return ((court.astype(int) * 10000 
                + year.astype(int)).astype(str) 
                + name)
    if dict_type == 2:
        return year.astype(int).astype(str) + name
    
def make_name_dicts() -> pd.DataFrame:
    names = []
    df = load_file(instructions.NAMES_PATH)
    df['name_perm'] = df['name_perm'].str.upper()
    df['concat_name1'] = convert_judge_name_series(
        df['year'], 
        df['court_num'], 
        df['name_perm'], 
        dict_type = 1)
    df['concat_name2'] = convert_judge_name_series(
        df['year'], 
        df['circuit_num'], 
        df['name_perm'], 
        dict_type = 1)
    df['concat_name3'] = convert_judge_name_series(
        df['year'], 
        df['court_num'], 
        df['name_perm'], 
        dict_type = 2)
    df['concat_name4'] = convert_judge_name_series(
        df['year'], 
        df['circuit_num'], 
        df['name_perm'], 
        dict_type = 2)
    names.append(names.set_index('concat_name1').to_dict()['judge_name'])
    names.append(names.set_index('concat_name2').to_dict()['judge_name'])
    names.append(names.set_index('concat_name3').to_dict()['judge_name'])
    names.append(names.set_index('concat_name4').to_dict()['judge_name'])
    names.append(names.set_index('name_perm').to_dict()['judge_name'])
    names.reset_index()
    names = names.drop_duplicates(ignore_index = True)
    return names

def main() -> None:
    warnings.filterwarnings('ignore')
    # The fjc website isn't working well with requests. So, you need to
    # manually download and rename for now.
    # download_fjc()
    service, career, demographics = load_fjc()
    df = combine_fjc(service = service, 
                     career = career, 
                     demographics = demographics)
    df = munge_fjc(df = df)
    df.to_csv(instructions.OUTPUT_PATH, index = False)
    names = make_names_df(df = df)
    names.to_csv(instructions.NAMES_PATH, index = False)
      
if __name__ == '__main__':
    main()
    