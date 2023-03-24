"""
@File : data_processing.py
@Author : 计科18-1 181002105 蒋政
@Date : 2021/7/7
@Task : 
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from imblearn import over_sampling
def data_preprocessing(train_data:pd.DataFrame, is_test=False):
    #1、缺失值处理->直接舍弃
    print("原始数据量：{}".format(len(train_data)))
    train_data.dropna(subset=['Driving_License','Policy_Sales_Channel'], inplace=True)
    print(train_data.isnull().sum())
    print("删除缺失值后数据量为:{}".format(len(train_data)))
    #离散值one-hot编码
    train_data = pd.get_dummies(train_data)
    train_data = train_data.rename(columns={"Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year", "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"})
    print(train_data.info())
    #连续值异常值处理（四分位距处理）
    print('Count of rows before filtering outlier: {}'.format(len(train_data)))
    filtered_entries = np.array([True] * len(train_data))
    for col in ['Annual_Premium']:
        Q1 = train_data[col].quantile(0.25)
        Q3 = train_data[col].quantile(0.75)
        IQR = Q3 - Q1
        low_limit = Q1 - (IQR * 1.5)
        high_limit = Q3 + (IQR * 1.5)

        filtered_entries = ((train_data[col] >= low_limit) & (train_data[col] <= high_limit)) & filtered_entries
    train_data = train_data[filtered_entries]
    train_label = train_data['Response']
    print(f'Count of rows after filtering outlier: {len(train_data)}')

    #标准化
    num_feat = ['Age', 'Vintage', 'Annual_Premium'] #连续特征
    cat_feat = ['Gender', 'Driving_License', 'Previously_Insured', 'Vehicle_Age_lt_1_Year', 'Vehicle_Age_gt_2_Years',
                'Vehicle_Damage', 'Region_Code', 'Policy_Sales_Channel'] #离散特征
    ss = StandardScaler()
    train_data[num_feat] = ss.fit_transform(train_data[num_feat])
    # mm = MinMaxScaler()
    # train_data[['Annual_Premium']] = mm.fit_transform(train_data[['Annual_Premium']])
    train_label = train_data['Response']
    train_data = train_data.drop(['id', 'Response'], axis=1)
    # for column in cat_feat:
    #     train_data[column] = train_data[column].astype('str')
    print(train_data.info())

    #解决样本不平衡问题
    # RandomOverSampler
    print(pd.Series(train_label).value_counts())
    x_over, y_over = over_sampling.RandomOverSampler().fit_resample(train_data, train_label)
    print(pd.Series(y_over).value_counts())
    return x_over, y_over
def data_preprocessing_test(train_data:pd.DataFrame):
    print(train_data.isnull().sum())
    train_data = train_data.fillna(method='bfill')
    # 离散值one-hot编码
    train_data = pd.get_dummies(train_data)
    train_data = train_data.rename(
        columns={"Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year", "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"})
    print(train_data.info())

    # 标准化
    num_feat = ['Age', 'Vintage', 'Annual_Premium']  # 连续特征
    cat_feat = ['Gender', 'Driving_License', 'Previously_Insured', 'Vehicle_Age_lt_1_Year', 'Vehicle_Age_gt_2_Years',
                'Vehicle_Damage', 'Region_Code', 'Policy_Sales_Channel']  # 离散特征
    ss = StandardScaler()
    train_data[num_feat] = ss.fit_transform(train_data[num_feat])
    train_data = train_data.drop(['id'], axis=1)
    print(train_data.info())
    return train_data
if __name__ == '__main__':
    # 读取数据
    train_data = pd.read_csv('./train.csv')
    # test_data = pd.read_csv('./test.csv')
    train = train_data.drop(['Response'], axis=1)
    train_label = train_data['Response']
    x_train, x_valid, y_train, y_valid = train_test_split(train_data, train_label, random_state=30, test_size=0.3)
    print(x_train.info())
    X, Y = data_preprocessing(x_train)
    #保存
    df = pd.concat([X, pd.DataFrame(Y).rename(columns={0: 'Response'})], axis=1)
    df.to_csv('train_processed.csv')
