import os
import numpy as np
import pandas as pd
import sys
from data_utils.base import *
from collections import Counter
from sklearn.preprocessing import StandardScaler


def PAMAP_dict():
    # all_actions = [
    #     "Lying: The act of reesting in a horizontal position, usually on a flat surface such as a bed or the ground.",
    #     "Sitting: Being in a seated position with the weight supported by the buttocks and thighs, often on a chair or bench.",
    #     "Standing: Maintaining an upright position on the feet without any external support.",
    #     "Walking: Moving at a regular pace by taking steps with alternate feet touching the ground.",
    #     "Running: Moving rapidly on foot by taking quick strides and often at a pace faster than walking.",
    #     "Cycling: Riding a bicycle, typically using pedals and a chain to propel oneself forward.",
    #     "Nordic Walking: A fitness activity that involves walking with the help of specially designed poles, engaging both the upper and lower body.",
    #     "Ascending Stairs: Moving upward by stepping on a series of elevated platforms or steps.",
    #     "Descending Stairs: Moving downward by stepping down from a series of elevated platforms or steps.",
    #     "Vacuum Cleaning: Using a vacuum cleaner to remove dirt, dust, and debris from floors and other surfaces.",
    #     "Ironing: Using a heated iron to remove wrinkles from clothing or fabric by pressing.",
    #     "Rope Jumping: Exercising by jumping over a rope that is swung over the head and under the feet continuously."
    # ]
    all_actions = ["Lying", "Sitting", "Standing", "Walking", "Running", "Cycling", "Nordic Walking",
                   "Ascending Stairs", "Descending Stairs", "Vacuum cleaning", "Ironing", "Rope Jumping"]

    action_idx_dict = {i: all_actions[i] for i in range(len(all_actions))}
    return action_idx_dict


def PAMAP(dataset_dir='./PAMAP2_Dataset/Protocol', WINDOW_SIZE=171, OVERLAP_RATE=0.5, SPLIT_RATE=(8, 2),
          VALIDATION_SUBJECTS={105}, Z_SCORE=True, SAVE_PATH=os.path.abspath('../../HAR-datasets')):
    '''
        dataset_dir: 源数据目录 : str
        WINDOW_SIZE: 滑窗大小 : int
        OVERLAP_RATE: 滑窗重叠率 : float in [0，1）
        SPLIT_RATE: 平均法分割验证集，表示训练集与验证集比例。优先级低于"VALIDATION_SUBJECTS": tuple
        VALIDATION_SUBJECTS: 留一法分割验证集，表示验证集所选取的Subjects : set
        Z_SCORE: 标准化 : bool
        SAVE_PATH: 预处理后npy数据保存目录 : str
    '''

    print(
        "\n原数据分析：共12个活动，文件包含9个受试者收集的数据，切分数据集思路可以采取留一法，选取n个受试者数据作为验证集。\n")
    print('预处理思路：提取有效列，重置活动label，数据降采样1/3，即100Hz -> 33.3Hz，进行滑窗，缺值填充，标准化等方法\n')

    #  保证验证选取的subjects无误
    if VALIDATION_SUBJECTS:
        print('\n---------- 采用【留一法】分割验证集，选取的subject为:%s ----------\n' % (VALIDATION_SUBJECTS))
        for each in VALIDATION_SUBJECTS:
            assert each in set([*range(101, 110)])
    else:
        print('\n---------- 采用【平均法】分割验证集，训练集与验证集样本数比为:%s ----------\n' % (str(SPLIT_RATE)))

    # 下载数据集
    # download_dataset(
    #     dataset_name='PAMAP2',
    #     file_url='http://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip',
    #     dataset_dir=dataset_dir
    # )

    xtrain, xtest, ytrain, ytest = [], [], [], []  # train-test-data, 用于存放最终数据
    # category_dict = dict(zip([*range(12)], [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]))  # 12分类所对应的实际label，对应readme.pdf

    category_dict = dict(zip([*range(12)], [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]))  # 12分类所对应的实际label，对应readme.pdf

    # category_dict = dict(zip([*range(18)], [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12 ,13, 16, 17, 18, 19, 20, 24]))  # 12分类所对应的实际label，对应readme.pdf


    dir = dataset_dir
    filelist = os.listdir(dir)
    os.chdir(dir)
    print('Loading subject data')
    for file in filelist:

        subject_id = int(file.split('.')[0][-3:])
        print('     current subject: 【%d】' % (subject_id), end='')
        print('   ----   Validation Data' if subject_id in VALIDATION_SUBJECTS else '')

        content = pd.read_csv(file, sep=' ', usecols=[1] + [*range(4, 16)] + [*range(21, 33)] + [
            *range(38, 50)])  # 取出有效数据列, 第2列为label，5-16，22-33，39-50都是可使用的传感数据
        content = content.interpolate(method='linear', limit_direction='forward', axis=0).to_numpy()  # 线性插值填充nan

        # 降采样 1/3， 100Hz -> 33.3Hz
        data = content[::3, 1:]  # 数据 （n, 36)
        label = content[::3, 0]  # 标签

        data = data[label != 0]  # 去除0类
        label = label[label != 0]

        for label_id in range(12):
            true_label = category_dict[label_id]
            cur_data = sliding_window(array=data[label == true_label], windowsize=WINDOW_SIZE, overlaprate=OVERLAP_RATE)

            # TODO: save data
            # 两种分割验证集的方法 [留一法 or 平均法]
            if VALIDATION_SUBJECTS:  # 留一法
                # 区分训练集 & 验证集
                if subject_id not in VALIDATION_SUBJECTS:  # 训练集
                    xtrain += cur_data
                    ytrain += [label_id] * len(cur_data)
                else:  # 验证集
                    xtest += cur_data
                    ytest += [label_id] * len(cur_data)
            else:  # 平均法
                trainlen = int(len(cur_data) * SPLIT_RATE[0] / sum(SPLIT_RATE))  # 训练集长度
                testlen = len(cur_data) - trainlen  # 验证集长度
                xtrain += cur_data[:trainlen]
                xtest += cur_data[trainlen:]
                ytrain += [label_id] * trainlen
                ytest += [label_id] * testlen

    os.chdir('../')

    xtrain = np.array(xtrain, dtype=np.float32)
    xtest = np.array(xtest, dtype=np.float32)
    ytrain = np.array(ytrain, np.int64)
    ytest = np.array(ytest, np.int64)

    if Z_SCORE:  # 标准化
        xtrain, xtest = z_score_standard(xtrain=xtrain, xtest=xtest)

    print(
        '\n---------------------------------------------------------------------------------------------------------------------\n')
    print('xtrain shape: %s\nxtest shape: %s\nytrain shape: %s\nytest shape: %s' % (
    xtrain.shape, xtest.shape, ytrain.shape, ytest.shape))

    if SAVE_PATH:  # 数组数据保存目录
        save_npy_data(
            dataset_name='PAMAP2',
            root_dir=SAVE_PATH,
            xtrain=xtrain,
            xtest=xtest,
            ytrain=ytrain,
            ytest=ytest
        )

    return xtrain, xtest, ytrain, ytest



def gen_PAMAP_data(dataset_dir='./PAMAP2_Dataset/Protocol', WINDOW_SIZE=171, OVERLAP_RATE=0.5, SPLIT_RATE=(8, 2),
          VALIDATION_SUBJECTS={105}, Z_SCORE=True):

    print(
        "\n原数据分析：共12个活动，文件包含9个受试者收集的数据，切分数据集思路可以采取留一法，选取n个受试者数据作为验证集。\n")
    print('预处理思路：提取有效列，重置活动label，数据降采样1/3，即100Hz -> 33.3Hz，进行滑窗，缺值填充，标准化等方法\n')

    if VALIDATION_SUBJECTS:
        print('\n---------- 采用【留一法】分割验证集，选取的subject为:%s ----------\n' % (VALIDATION_SUBJECTS))
        for each in VALIDATION_SUBJECTS:
            assert each in set([*range(101, 110)])
    else:
        print('\n---------- 采用【平均法】分割验证集，训练集与验证集样本数比为:%s ----------\n' % (str(SPLIT_RATE)))


    xtrain, xtest, ytrain, ytest = [], [], [], []  # train-test-data, 用于存放最终数据
    all_data, all_label = [], []
    category_dict = dict(zip([*range(12)], [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]))  # 12分类所对应的实际label，对应readme.pdf
    # category_dict = dict(zip([*range(18)], [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12 ,13, 16, 17, 18, 19, 20, 24]))  # 12分类所对应的实际label，对应readme.pdf

    dir = dataset_dir
    filelist = os.listdir(dir)
    print('Loading subject data')
    for file_path in filelist:
        file = os.path.join(dir, file_path)
        subject_id = int(file.split('.')[0][-3:])
        print('     current subject: 【%d】' % (subject_id), end='')
        print('   ----   Validation Data' if subject_id in VALIDATION_SUBJECTS else '')

        content = pd.read_csv(file, sep=' ', usecols=[1] + [*range(4, 16)] + [*range(21, 33)] + [
            *range(38, 50)])  # 取出有效数据列, 第2列为label，5-16，22-33，39-50都是可使用的传感数据
        content = content.interpolate(method='linear', limit_direction='forward', axis=0).to_numpy()  # 线性插值填充nan

        data = content[::3, 1:]  # 数据) （n, 36)
        label = content[::3, 0]  # 标签

        data = data[label != 0]  # 去除0类
        label = label[label != 0]

        for label_id in range(12):
            true_label = category_dict[label_id]
            cur_data = sliding_window(array=data[label == true_label], windowsize=WINDOW_SIZE, overlaprate=OVERLAP_RATE)

            all_data += cur_data
            all_label += [label_id] * len(cur_data)

    all_data = np.array(all_data, dtype=np.float32)
    all_label = np.array(all_label, np.int64)


    if Z_SCORE:  # 标准化
        all_data = z_score_standard_single(all_data)

    print(
        '\n---------------------------------------------------------------------------------------------------------------------\n')
    print('all_data shape: %s\nall_label shape: %s' % (
    all_data.shape, all_label.shape))

    np.save('./data_utils/pamap_data.npy', all_data)
    np.save('./data_utils/pamap_label.npy', all_label)


loc = [1] + [*range(4, 16)] + [*range(21, 33)] + [*range(38, 50)]

def window(data, label, size, stride):
    '''将数组data和label按照滑窗尺寸size和stride进行切割'''
    x, y = [], []
    for i in range(0, len(label), stride):
        if i+size < len(label): #不足一个滑窗大小的数据丢弃

            l = set(label[i:i+size])
            if len(l) > 1 or label[i] == 0: #当一个滑窗中含有包含多种activity或者activity_id为0（即属于其他动作），丢弃
                continue
            elif len(l) == 1:
                x.append(data[i: i + size, :])
                y.append(label[i])

    return x, y

def gen_PAMAP_data2(dataset_dir, window_size, step):
    X, Y = [], []
    # 遍历9个subject文件
    for i in range(1, 10):
        total = pd.read_csv(f'{dataset_dir}/subject10' + str(i) + '.dat', header=None, sep=' ', usecols=loc).values
        total = total[~np.isnan(total).any(axis=1), :]  # 去除NaN
        data = total[:, 1:]
        label = total[:, 0].reshape(-1)

        # 调用window函数进行滑窗处理
        x, y = window(data, label, window_size, step)
        X += x
        Y += y

    # 将索引从0开始依次编号
    cate_idx = list(Counter(Y).keys())
    cate_idx.sort()
    for i in range(len(Y)):
        Y[i] = cate_idx.index(Y[i])
    X = np.array(X)
    Y = np.array(Y)
    X = z_score_standard_single(X)

    print(f"PAMAP2 data shape: {X.shape}, label shape: {Y.shape}")
    np.save('./data_utils/pamap_data.npy', X)
    np.save('./data_utils/pamap_label.npy', Y)



if __name__ == '__main__':
    # PAMAP(dataset_dir="/home/dingding/Datasets/PAMAP2_Dataset/Protocol")
    # gen_PAMAP_data(dataset_dir="/home/dinghao/Dataset/PAMAP2_Dataset/Protocol/")
    dataset_dir = "/home/dingding/Datasets/PAMAP2_Dataset/Protocol"
    # x, y = gen_PAMAP_data2(dataset_dir, 171, 85)
    # x = np.array(x)
    # y = np.array(y)
    # print(x.shape, y.shape)