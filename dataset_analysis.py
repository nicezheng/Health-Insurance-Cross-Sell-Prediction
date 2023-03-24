"""
@File : dataset_analysis.py
@Author : 计科18-1 181002105 蒋政
@Date : 2021/7/7
@Task : 
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def datainfo_analysis(data:pd.DataFrame):
    #数据表头
    print(data.head())
    #数据信息（数据结构，数据量）
    print(data.info())
    #数据缺失值
    print(data.isnull().sum())
    #数据重复值
    print(data.duplicated().sum())
def data_visualization(train:pd.DataFrame, is_test=False):
    #离散值分析Categorical variables analysis
    fig, ax = plt.subplots(2, 3, figsize=(30, 20))
    # Gender plot
    g_gender = sns.countplot(data=train, x='Gender', hue = train['Gender'],palette=sns.color_palette('Set2'), ax=ax[0, 0])

    # Driving License plot
    g_driving_license = sns.countplot(data=train, x='Driving_License', hue = train['Driving_License'],palette=sns.color_palette('Set2'), ax=ax[1, 0])

    # Previously insured plot
    g_previously_insured = sns.countplot(data=train, x='Previously_Insured', hue = train['Previously_Insured'],palette=sns.color_palette('Set2'), ax=ax[0, 1])

    # Vehicle damage plot
    g_damage = sns.countplot(data=train, x='Vehicle_Damage', hue = train['Vehicle_Damage'],palette=sns.color_palette('Set2'), ax=ax[1, 1])

    # Vehicle Age plot
    g_vehicle_age = sns.countplot(data=train, x='Vehicle_Age', hue = train['Vehicle_Age'],palette=sns.color_palette('Set2'), ax=ax[0, 2])

    # Response plot
    if not is_test:
        g_response = sns.countplot(data=train, x='Response', hue = train['Response'],palette=sns.color_palette('Set2'), ax=ax[1, 2])

    # Titles
    ax[0, 0].set_title('Gender', fontsize=20)
    ax[1, 0].set_title('Driving License', fontsize=20)
    ax[0, 1].set_title('Previously Insured', fontsize=20)
    ax[1, 1].set_title('Vehicle Damage', fontsize=20)
    ax[0, 2].set_title('Vehicle Age', fontsize=20)
    if not is_test:
        ax[1, 2].set_title('Response', fontsize=20)

    # Delete x and y labels
    for ax in ax.reshape(-1):
        ax.set_xlabel(None)
        ax.set_ylabel(None)

    # Super title
    # fig.suptitle('', size='40', y=1.0)

    plt.tight_layout()  # Plots fit the fig area
    plt.show()
    plt.close()

    #连续值分析Continuous variables analysis
    fig, ax = plt.subplots(5, 1, figsize=(30, 20))
    # Age plot
    g_edad = sns.histplot(data=train, x='Age', kde=True, ax=ax[0])

    # Region Code plot
    g_region = sns.histplot(data=train, x='Region_Code', kde=True, ax=ax[1])

    # Premium annual plot
    g_premium_anual = sns.histplot(data=train, x='Annual_Premium', kde=True, ax=ax[2])

    # Policy sales channel plot
    g_policy_sales_channel = sns.histplot(data=train, x='Policy_Sales_Channel', kde=True, ax=ax[3])

    # Vintage plot
    g_vintage = sns.histplot(data=train, x='Vintage', kde=True, ax=ax[4])

    # Titles
    ax[0].set_title('Age', fontsize=20)
    ax[1].set_title('Region Code', fontsize=20)
    ax[2].set_title('Annual Premium', fontsize=20)
    ax[3].set_title('Policy Sales Channel', fontsize=20)
    ax[4].set_title('Vintage', fontsize=20)

    # Set x ticks limit to premium annual due to its nonuniform distribution
    ax[2].set_xlim(0, 90000)

    # Delete x and y label
    for ax in ax.reshape(-1):
        ax.set_xlabel(None)
        ax.set_ylabel(None)

    # Super title
    # fig.suptitle(, size='40', y=1.0)

    plt.tight_layout()  # Plots fit the fig area
    plt.show()
    plt.close()
    #多值比较分析Variables comparison

if __name__ == '__main__':
    # 读取数据
    train_data = pd.read_csv('./train.csv')
    test_data = pd.read_csv('./test.csv')
    # train = train_data.drop(['Response'], axis=1)
    # train_label = train_data['Response']
    # x_train, x_valid, y_train, y_valid = train_test_split(train_data, train_label, random_state=30, test_size=0.3)
    datainfo_analysis(train_data)
    data_visualization(train_data)
    datainfo_analysis(test_data)
    data_visualization(test_data, is_test=True)