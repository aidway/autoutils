import subprocess
from hdfs.ext.kerberos import KerberosClient
import os
import configparser


FILE_BASE_PATH=__file__

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_active_namenode(namenode1, namenode2):
    '''
    功能：获取active namenode结点
    '''
    try:
        #status = subprocess.call(['kinit -kt ./hdfs.keytab  ' + tdh_principal], shell=True)
        hdfs_client = KerberosClient(namenode1, principal=tdh_principal)
        tmp_dir_status = hdfs_client.status('/')
        active_namenode = namenode1
    except:
        active_namenode = namenode2
    return active_namenode

def get_hdfs_client(active_namenode, tdh_principal):
    '''
    功能：获取hdfs client
    '''
    hdfs_client = KerberosClient(active_namenode, principal=tdh_principal)
    return hdfs_client

def hdfs_ls(path):
    '''
    功能：ls
    path： hdfs路径
    '''
    return hdfs_client.list(path)

def hdfs_put(target_path, source_file):
    '''
    功能：put
    target_path： hdfs路径
    source_file： 本地待上传文件
    '''
    hdfs_client.upload(target_path, source_file)
    
def hdfs_download(source_file, target_path):
    '''
    功能：ls
    source_file： hdfs待下载文件
    target_path： 存储下载文件的目标路径
    '''
    hdfs_client.download(source_file, target_path)
    
def hdfs_delete(target_file):
    '''
    功能：ls
    target_file： hdfs路径上待删除的文件
    '''
    hdfs_client.delete(target_file)


cf = configparser.ConfigParser()
cf.read(os.path.join(BASE_DIR, 'conf', 'hdfsutils.conf'))
namenode1 = cf.get('hdfs', 'namenode1')
namenode2 = cf.get('hdfs', 'namenode2')
tdh_principal = cf.get('hdfs', 'tdh_principal')
hdfs_keytab_path = os.path.join(BASE_DIR, 'conf', 'hdfs.keytab')
    
# kinit    
#status = subprocess.call(['kinit -kt ./hdfs.keytab  ' + tdh_principal], shell=True)
status = subprocess.call(['kinit -kt ' + hdfs_keytab_path + ' ' + tdh_principal], shell=True)

active_namenode = get_active_namenode(namenode1, namenode2)
hdfs_client = get_hdfs_client(active_namenode, tdh_principal)

# if __name__ == '__main__':
#     cf = configparser.ConfigParser()
#     cf.read(os.path.join(BASE_DIR, 'conf', 'hdfsutils.conf'))
#     namenode1 = cf.get('hdfs', 'namenode1')
#     namenode2 = cf.get('hdfs', 'namenode2')
#     tdh_principal = cf.get('hdfs', 'tdh_principal')
    
#     active_namenode = get_active_namenode(namenode1, namenode2)
#     hdfs_client = get_hdfs_client(active_namenode, tdh_principal)
    
#     hdfs_put('/tmp/jinan_subway', './hdfs.keytab')
#     print(hdfs_ls('/tmp/jinan_subway'))
#     hdfs_download('/tmp/jinan_subway/hdfs.keytab', './tmp')
#     hdfs_delete('/tmp/jinan_subway/hdfs.keytab')
#     print('active_namenode:', active_namenode)