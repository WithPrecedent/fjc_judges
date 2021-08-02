"""
fjc_match: functions for using fjc judge data for name matching
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2020-2021, Corey Rayburn Yung
License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import annotations

from judges import fjc_create
from judges import instructions


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
    
def make_name_dicts() -> list[dict[str, str]]:
    """Converts name perms with other data to a single mapping.
    
    This creates a very ugly set of matching names with years and courts mixed
    in. For it to work, you need to similarly convert names in the court data
    DataFrame. The reason to use this is pure speed over a large dataset. 
    Instead of doing a cross-walk over multiple columns, you can use a python
    dict on a single column. With a lot of data, this is ugly, but much 
    faster.
    
    """
    names = []
    df = fjc_create.load_file(instructions.NAMES_PATH)
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
    names.append(df.set_index('concat_name1').to_dict()['judge_name'])
    names.append(df.set_index('concat_name2').to_dict()['judge_name'])
    names.append(df.set_index('concat_name3').to_dict()['judge_name'])
    names.append(df.set_index('concat_name4').to_dict()['judge_name'])
    names.append(df.set_index('name_perm').to_dict()['judge_name'])
    return names

def main() -> None:
    names = make_name_dicts()
    return names
    
if __name__ == '__main__':
    main()
    