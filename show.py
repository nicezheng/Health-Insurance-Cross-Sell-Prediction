"""
@File : show.py
@Author : 计科18-1 181002105 蒋政
@Date : 2021/7/4
@Task : 
"""
import seaborn as sns
import pandas as pd
import numpy as np
def lisanhua(data, column):
    df = data.copy()
    data_set = set(df[:,column])
    data_dict = {}
    for i,item in enumerate(data_set):
        data_dict[item] = i
    for i in range(len(df[column])):
        df[column][i] = data_dict[df[column][i]]
    return df

#读取数据
train_data = pd.read_csv('./minitrain.csv')
#数据预处理
#1、性别 离散二值化
gender = train_data['Gender'].values
train_data = lisanhua(train_data, 'Gender')
print(train_data['Gender'].values)
age = train_data['Age'].values
driving_license = train_data['Driving_License']
region_Code = train_data['Region_Code']
previously_Insured = train_data['Previously_Insured']
vehicle_Age = train_data['Vehicle_Age']
vehicle_Damage = train_data['Vehicle_Damage']
annual_Premium = train_data['Annual_Premium']
policy_Sales_Channel = train_data['Policy_Sales_Channel']
vintage = train_data['Vintage']
label = train_data['Response']
print(gender)
sns.set_style('darkgrid')
sns.distplot(gender)
print(age)