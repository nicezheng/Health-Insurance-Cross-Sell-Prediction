"""
@File : data_analysis.py
@Author : 计科18-1 181002105 蒋政
@Date : 2021/7/4
@Task : 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
def data_analysis(train_data:pd.DataFrame, test_data:pd.DataFrame):
    print(train_data.head())
    print(test_data.head())
    print("训练集：{}， 测试集：{}".format(str(train_data.shape), str(test_data.shape)))
    print(train_data.isnull().sum())
    #TODO 缺失值处理
    numerical_columns = ['Age', 'Region_Code', 'Annual_Premium', 'Vintage']
    categorical_columns = ['Gender', 'Driving_License', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage',
                           'Response']
    print(train_data[numerical_columns].describe())




def solve(x,y):
    #划分数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=30)
    #boosting
    import xgboost as xgb
    import catboost as cb
    import lightgbm as lgb
    #
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import VotingClassifier
    from sklearn.metrics import roc_auc_score,classification_report,accuracy_score, roc_curve
    from sklearn.linear_model import LogisticRegression
    from matplotlib.pyplot import plot,title,xlabel,ylabel
    import datetime
    from sklearn import metrics
    LR = LogisticRegression(C=0.1, solver='newton-cg',random_state=30)
    Ada = AdaBoostClassifier(algorithm="SAMME", learning_rate=0.1, n_estimators=100, random_state=30)
    GBDT = GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse',learning_rate=0.7, loss='exponential',random_state=30,max_features='auto')
    svc = SVC(C=0.8, gamma=20, random_state=30)
    rf = RandomForestClassifier(random_state=30)
    xgb = xgb.XGBClassifier(random_state=30)
    weights = []
    estimators_list = [('LR', LR), ('Ada', Ada), ('GBDT', GBDT), ('SVC', svc), ('rf', rf)]

    for label, clf in estimators_list:
        start_time = datetime.datetime.now()
        clf.fit(x_train, y_train)
        end_time = datetime.datetime.now()
        print("训练时间：", (end_time-start_time))
        y_pred = clf.predict(x_test)
        print("{}在训练集上分类结果".format(label))
        print(classification_report(y_train, y_pred=clf.predict(x_train)))
        print("{}在测试集上分类结果".format(label))
        print(classification_report(y_test,y_pred=y_pred))
        print('{}的ROC面积为'.format(label), roc_auc_score(y_test, y_pred))
        score = accuracy_score(y, clf.predict(x))
        weights.append(score)
    #软投票
    w = weights /sum(weights)
    vote2 = VotingClassifier(estimators=estimators_list, voting='soft', weights=w)
    vote2.fit(x_train, y_train)
    y_pred = vote2.predict(x_test)
    print("{}在测试机上的分类结果：".format('soft Voting'))
    print(classification_report(y_test, y_pred))
    #画图
    print('{}的ROC面积为'.format('soft Voting'), roc_auc_score(y_test, y_pred))
    y_score = vote2.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    title('Random Forest ROC curve: CC Fraud')
    xlabel('FPR (Precision)')
    ylabel('TPR (Recall)')
    plot(fpr, tpr)
    plot((0, 1), ls='dashed', color='black')
    plt.show()


if __name__ == '__main__':
    #读取数据
    train_data = pd.read_csv('./train.csv')
    # val = pd.read_csv('./valid.csv')
    test_data = pd.read_csv('./test.csv')
    #数据分析
    # data_analysis(train_data, test_data)
    # show_var_analysis(train_data, test_data)
    train_data = data_preprocessing(train_data)
    train = train_data.drop(['Response'], axis=1)
    train_target = train_data['Response']
    print("train列名",list(train))
    print("train_target列名", list(train_target))
    solve(train, train_target)
