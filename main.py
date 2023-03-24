"""
@File : main.py
@Author : 计科18-1 181002105 蒋政
@Date : 2021/7/7
@Task : 
"""
import datetime
import pickle

from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import pandas as pd
from dataset_processing import data_preprocessing, data_preprocessing_test
from model import get_model_params
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
from matplotlib.pyplot import title,xlabel,ylabel,plot
import sys
# sys.stdout = open('print.log', mode = 'w',encoding='utf-8')
def show_auc(clf, X, Y, model=None):
    y_score = clf.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(Y, y_score)
    # plt.figure(0).clf()
    title('ROC curve'.format(model))
    xlabel('FPR (Precision)')
    ylabel('TPR (Recall)')
    print('{}(area = {})'.format(model, str(auc(fpr, tpr))))
    plot(fpr, tpr,label='{}(area = {})'.format(model, str(auc(fpr, tpr))))
    plot((0,1) ,ls='dashed', color='black')
    plt.legend(loc=0)  # 说明所在位置
    plt.savefig('{}.png'.format(model))
    with open('{}_log.log'.format(model), 'w+') as f:
        f.write('Area under curve (AUC): {}\n'.format(auc(fpr, tpr)))
    print('Area under curve (AUC): ', auc(fpr, tpr))
def save_df(split, x,y):
    df = pd.concat([x, pd.DataFrame(y).rename(columns={0: 'Response'})], axis=1)
    df.to_csv('{}_processed.csv'.format(split))  # 保存训练数据
def df_cat(x,y):
    return pd.concat([x, pd.DataFrame(y).rename(columns={0: 'Response'})], axis=1)

if __name__ == '__main__':
    # 读取数据
    train_data = pd.read_csv('./minitrain.csv')

    train = train_data.drop(['Response'], axis=1)
    train_label = train_data['Response'].astype(int)
    x_train, x_valid, y_train, y_valid = train_test_split(train, train_label, random_state=30, test_size=0.2)
    print(x_train.info())
    #X,Y
    X, Y = data_preprocessing(df_cat(train, train_label))
    # save_df('train_all', X, Y)  # 保存训练数据
    #train
    X_train, Y_train = data_preprocessing(df_cat(x_train, y_train))
    # save_df('train', X_train, Y_train)#保存训练数据
    X_valid, Y_valid = data_preprocessing(df_cat(x_valid, y_valid))
    # save_df('valid', X_valid, Y_valid)  # 保存训练数据
    #test
    test_data = pd.read_csv('./minitest.csv')
    X_test = data_preprocessing_test(test_data)
    clf_list = ['lgb','rf','cb','knn']
    for clf_name in clf_list:
        print('-'*100, clf_name, '-'*100)
        model,params_dict = get_model_params(clf_name)
        #train
        start_time = datetime.datetime.now()
        model.fit(X_train, Y_train)
        end_time = datetime.datetime.now()
        print("训练时间：", (end_time - start_time))
        print("{}在训练集上分类结果".format(clf_name))
        print(classification_report(Y_train, y_pred=model.predict(X_train)))
        #valid调参
        grid = RandomizedSearchCV(model, params_dict, cv=5, scoring='roc_auc', n_iter=20, n_jobs=-1)
        start_time = datetime.datetime.now()
        grid.fit(X_valid, Y_valid)
        end_time = datetime.datetime.now()
        print(">>>>验证消耗时间：", (end_time - start_time))
        print('>>>>best score:{}'.format(grid.best_score_))#显示best超参数
        print('>>>>best parameters:')
        for key in params_dict.keys():
            print('{}:{}'.format(key, grid.best_estimator_.get_params()[key]))
        print("{}在验证集上分类结果".format(clf_name))
        Y_pred = grid.predict(X_valid)
        show_auc(grid, X_valid, Y_valid, clf_name)
        print(classification_report(Y_valid, y_pred=Y_pred))

        best_hyperparams = grid.best_estimator_.get_params()
        pickle.dump(best_hyperparams, open('{}_minimodel_params.pkl'.format(clf_name), 'wb'))  # 保存模型
        #测试
        model,_ = get_model_params(clf_name, best_hyperparams)
        model.fit(X, Y)
        Preds = [pred[1] for pred in model.predict_proba(X_test)]
        id = test_data.id
        submission = pd.DataFrame(data={'id': id, 'Response': Preds})
        submission.to_csv('results_mini{}.csv'.format(clf_name), index=False)
        submission.head()
        # #predict
        # model, _ = get_model_params(clf_name, best_hyperparams)
        # model.fit(X, Y)
        # predict(model)