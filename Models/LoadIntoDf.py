import sqlite3
import pandas as pd

def load_sqlite_data(rootPath = '../../'):
    
    conn = sqlite3.connect(f'{rootPath}data/boardlib-kilter-db.sqlite3')
    query = "SELECT * FROM climbs_cleaned"

    df = pd.read_sql_query(query, conn)
    conn.close()

    df.set_index('uuid', inplace=True) 
    return df

def load_sqlite_data_smaller(min_ascentionist_count=3):
    
    conn = sqlite3.connect('../../data/boardlib-kilter-db.sqlite3')
    query = f"""select cc.* 
    from climbs_cleaned cc 
    join climb_stats cs on cs.climb_uuid == cc.uuid and cc.angle == cs.angle 
    where cs.ascensionist_count > {min_ascentionist_count}"""
    df = pd.read_sql_query(query, conn)
    conn.close()

    df.set_index('uuid', inplace=True) 
    return df



def load_asc_count():
    
    conn = sqlite3.connect('../data/boardlib-kilter-db.sqlite3')
    query = f"""select cs.ascensionist_count, count(*)
    from climbs_cleaned cc 
    join climb_stats cs on cs.climb_uuid == cc.uuid and cc.angle == cs.angle
    group by cs.ascensionist_count"""
    df = pd.read_sql_query(query, conn)
    conn.close()

    return df
