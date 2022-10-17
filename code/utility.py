import os
import time
import sys
import datetime

def timeStamp(return_second=False):
    t = datetime.datetime.fromtimestamp(time.time())
    if return_second:
        return '{}{}{}_{}{}{}'.format(t.year, t.month, t.day, t.hour, t.minute, t.second)
    else:
        return '{}{}{}_{}{}'.format(t.year, t.month, t.day, t.hour, t.minute)
    
def AssertTerminate(equation, msg=None):
    if not equation:
        print(msg)
        sys.exit(1)

def MakeLogFP(fname, add_timestamp=False):
    if add_timestamp:
        fname += '_{}.log'.format(timeStamp())

    fp = open(fname, "w", 1) # line bufffering
    return fp

def WriteMsgLogStdout(fp=None, msg='',  supress_std_out=False, add_newline=True):
    if add_newline:
        msg +='\n'
    if not supress_std_out:
        print(msg)

    if hasattr(fp, 'read'): # file object testing
        if not isinstance(msg, str): # converting to string
            msg = str(msg)
        fp.write(msg)

def search_exist_suffix(f_path):

    for i in range(1,1000):
        n_name = f_path + str(i)
        if not os.path.exists(n_name):
            return i
    else:
        ValueError('cannot find unused folder')

def MakeFolder(folder_path, allow_override=False, skip_create=False):
    """
    Make a folder
    """
    if os.path.exists(folder_path) and skip_create:
        return folder_path

    if os.path.exists(folder_path) and (not allow_override):
        Warning('Specified folder already exists. Create new one')
        sufix = search_exist_suffix(folder_path)
        folder_path = folder_path + str(sufix)

    if not allow_override or not os.path.exists(folder_path):
        os.mkdir(folder_path)

    return folder_path
