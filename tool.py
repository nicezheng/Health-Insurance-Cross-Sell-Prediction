"""
@File : tool.py
@Author : 计科18-1 181002105 蒋政
@Date : 2021/7/4
@Task : 
"""
import pandas as pd
def split_minidataset(path, splitnum=1000):
    data = pd.read_csv('./{}.csv'.format(path))
    minidata = data[:splitnum]
    print(data[:splitnum])
    minidata.to_csv('./mini{}.csv'.format(path), index=False)
if __name__ == '__main__':
    split_minidataset('test')
    split_minidataset('train')
