from pyhive import hive
import pandas as pd
import configparser
import os
import pyodbc
from impala.dbapi import connect
from impala.util import as_pandas


import psycopg2
import time
import pymysql

FILE_BASE_PATH=__file__

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class DBConfig:
    cf = configparser.ConfigParser()
    cf.read(os.path.join(BASE_DIR, 'conf', 'dbutils.conf'))
#     cf.read(os.path.abspath('.') + '\conf\dbutils.conf')
    
    Driver = cf.get('inceptor', 'Driver')
    Host = cf.get('inceptor', 'Host')
    Hive = cf.get('inceptor', 'Hive')
    Port = cf.getint('inceptor', 'Port')
    User = cf.get('inceptor', 'User')
    Password = cf.get('inceptor', 'Password')
    Database = cf.get('inceptor', 'Database')
    Mech  = cf.get('inceptor', 'Mech')
    INCEPTOR_CONFIG_IMPYLA = dict(host=Host,port=int(Port),user=User,password=Password,database=Database,auth_mechanism='PLAIN')
    INCEPTOR_CONFIG_PYHIVE = hive.Connection(host=Host,port=int(Port),auth=Mech,database=Database,username=User,password=Password)
    INCEPTOR_CONFIG_PYHIVE_LOCAL = hive.Connection(host=Host,port=int(Port),auth=Mech,database=Database,username=User,password=Password,configuration={'ngmr.exec.mode':'LOCAL'})
    INCEPTOR_CONFIG_PYODBC = "Driver=%s;Server=%s;Hive=%s;Host=%s,Port=%s;Database=%s;User=%s;Password=%s;Mech=%s" %(Driver, Host, Hive, Host, Port, Database, User, Password, Mech)
    
    

def execSqlImpyla(sql,  INCEPTOR_CONFIG_IMPYLA=DBConfig.INCEPTOR_CONFIG_IMPYLA):
    '''
    功能：使用Impyla访问Inceptor
    返回：数据集，DataFrame格式
    '''
    
    con = connect(**INCEPTOR_CONFIG_IMPYLA)
    cur = con.cursor()
    cur.execute(sql)
    df_data = as_pandas(cur)
    cur.close()
    return df_data
    
    
def execSqlPyhive(sql, mode='cluster'):
    '''
    功能：使用pyhive访问Inceptor
    返回：数据集，DataFrame格式
    '''
    if mode == 'cluster':
        INCEPTOR_CONFIG_PYHIVE=DBConfig.INCEPTOR_CONFIG_PYHIVE
    elif mode == 'local':
        INCEPTOR_CONFIG_PYHIVE=DBConfig.INCEPTOR_CONFIG_PYHIVE_LOCAL
    else:
        print('Error mode')
        return -1
        
    cur = INCEPTOR_CONFIG_PYHIVE.cursor()      
    cur.execute(sql)
    columns = [col[0] for col in cur.description]
    result = [dict(zip(columns, row)) for row in cur.fetchall()]
    data = pd.DataFrame(result)
    data.columns = columns 
    cur.close()
    return data
    
def execSqlPyodc(sql, INCEPTOR_CONFIG_PYODBC=DBConfig.INCEPTOR_CONFIG_PYODBC):
    '''
    功能：使用pyodbc访问Inceptor
    返回：数据集，DataFrame格式
    '''
    db = pyodbc.connect(INCEPTOR_CONFIG_PYODBC)
    cursor = db.cursor()
    cursor.execute(sql)
    columns = [col[0] for col in cursor.description]
    result = [dict(zip(columns, row)) for row in cursor.fetchall()]
    data = pd.DataFrame(result)
    data.columns = columns 
    cursor.close()
    return data

def psycopg2gp(conn, sql):
   start = time.time()
   cur = conn.cursor()
   row = cur.execute(sql)
   header = [col[0] for col in cur.description]
   res = pd.DataFrame(list(cur.fetachall()), columns=header)
   print(f'fetch {row} rows')
   cost = time.time() - start 
   print(f'cost {cost: .3f}s')
   del cur
   return res
   
   
def pymysql2mysql(conn, sql):
   start = time.time()
   cur = conn.cursor()
   row = cur.execute(sql)
   header = [col[0] for col in cur.description]
   res = pd.DataFrame(list(cur.fetachall()), columns=header)
   print(f'fetch {row} rows')
   cost = time.time() - start 
   print(f'cost {cost: .3f}s')
   del cur
   return res
    
#if __name__ == '__main__':
#    INCEPTOR_CONFIG_IMPYLA = DBConfig.INCEPTOR_CONFIG_IMPYLA
#