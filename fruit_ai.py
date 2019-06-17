import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.model_selection import train_test_split

DATA_FILE = 'fruit_data.csv'

fruit_name = ['apple',
              'mandarin',
              'orange',
              'lemon']

Feat_cols = ['mass',
             'width',
             'height',
             'color_score']

def get_pred_fruit(test_sample_feat, train_data):
    dis_list = []
    for idx, row in train_data.iterrows():
        # 训练样本特征
        train_sample_feat = row[Feat_cols].values
        dis = euclidean(test_sample_feat, train_sample_feat)
        dis_list.append(dis)
    pos = np.argmin(dis_list)
    pred_fruit = train_data.iloc[pos]['fruit_name']
    return pred_fruit

def main():
    # 读取数据集
    fruit_data = pd.read_csv(DATA_FILE, index_col='id')

    # 划分数据集
    train_data, test_data = train_test_split(fruit_data, test_size=1/3, random_state=10)

    # 计数器
    account = 0

    # 分类器
    for idx, row in test_data.iterrows():
        # 测试样本特征
        test_sample_feat = row[Feat_cols].values

        # 预测
        pred_fruit = get_pred_fruit(test_sample_feat, train_data)

        #真实值
        true_fruit = row['fruit_name']

        print('样本{}的真实标签是{}，预测标签是{}'.format(idx, true_fruit, pred_fruit))

        if true_fruit == pred_fruit:
            account += 1

    accuracy = account / test_data.shape[0]

    print('预测准确率为{:.2f}%'.format(accuracy * 100))

if __name__=="__main__":
    main()
