static_unknown_text = "Unknown action"

import shutil
import os
import glob
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler


def merge_txt_files(file1, file2, output_file):
    with open(file1, 'r') as f1, open(file2, 'r') as f2, open(output_file, 'w') as out:
        content1 = f1.read()
        content2 = f2.read()

        out.write(content1)
        out.write(content2)

def add_unknown_class_to_txt_file(file, output_file):
    with open(file, 'r') as f, open(output_file, 'w') as out:
        content = f.read()
        out.write(content)
        out.write(static_unknown_text)

def download_dataset(dataset_name, file_url, dataset_dir):
    if os.path.exists(dataset_dir):
        return

    print(
        '\n==================================================【 %s Dataset Downloads】===================================================\n' % (
            dataset_name))
    print('url: %s\n' % (file_url))

    dir_path = dataset_dir.split('/')[0]
    if dataset_name == 'UniMiB-SHAR' and file_url[-4:] == '.git':
        if os.path.exists(os.path.join(dir_path, dataset_name)):
            shutil.rmtree(os.path.join(dir_path, dataset_name))
        os.system('git clone %s %s/%s' % (file_url, dir_path, dataset_name))

    else:
        # download
        dataset_file_path = os.path.join(dir_path, 'dataset.zip')
        os.system(f"wget -O {dataset_file_path} {file_url}")

        # unpack
        while glob.glob(os.path.join(dir_path, '*.zip')):
            for file in glob.glob(os.path.join(dir_path, '*.zip')):
                for format in ["zip", "tar", "gztar", "bztar", "xztar"]:
                    try:
                        shutil.unpack_archive(filename=file, extract_dir=dir_path, format=format)
                        break
                    except:
                        continue
                os.remove(file)

    print()

    if not os.path.exists(dataset_dir):
        quit('Datasets download failed, please check the url or network connection!')


def sliding_window(array, windowsize, overlaprate):
    stride = int(windowsize * (1 - overlaprate))
    times = (len(array) - windowsize) // stride + 1
    res = []
    for i in range(times):
        x = array[i * stride: i * stride + windowsize]
        res.append(x)
    return res


def z_score_standard(xtrain, xtest):
    assert xtrain.shape[1:] == xtest.shape[1:]
    window_size, modal_leng = xtrain.shape[1:]
    xtrain_2d, xtest_2d = xtrain.reshape(-1, modal_leng), xtest.reshape(-1, modal_leng)  # reshape成2维，按模态轴进行标准化
    std = StandardScaler().fit(xtrain_2d)
    xtrain_2d, xtest_2d = std.transform(xtrain_2d), std.transform(xtest_2d)
    xtrain, xtest = xtrain_2d.reshape(xtrain.shape[0], window_size, modal_leng), xtest_2d.reshape(xtest.shape[0],
                                                                                                  window_size,
                                                                                                  modal_leng)
    return xtrain, xtest

def z_score_standard_single(xtrain):
    window_size, modal_leng = xtrain.shape[1:]
    xtrain_2d = xtrain.reshape(-1, modal_leng)
    std = StandardScaler().fit(xtrain_2d)
    xtrain_2d = std.transform(xtrain_2d)
    xtrain = xtrain_2d.reshape(xtrain.shape[0], window_size, modal_leng)
    return xtrain


def build_npydataset_readme(path):
    datasets = sorted(os.listdir(path))
    curdir = os.curdir
    os.chdir(path)
    with open('readme.md', 'w') as w:
        for dataset in datasets:
            if not os.path.isdir(dataset):
                continue
            x_train = np.load('%s/x_train.npy' % (dataset))
            x_test = np.load('%s/x_test.npy' % (dataset))
            y_train = np.load('%s/y_train.npy' % (dataset))
            y_test = np.load('%s/y_test.npy' % (dataset))
            category = len(set(y_test.tolist()))
            d = Counter(y_test)
            new_d = {}
            for i in range(category):
                new_d[i] = d[i]
            log = '\n===============================================================\n%s\n   x_train shape: %s\n   x_test shape: %s\n   y_train shape: %s\n   y_test shape: %s\n\n共【%d】个类别\ny_test中每个类别的样本数为 %s\n' % (
            dataset, x_train.shape, x_test.shape, y_train.shape, y_test.shape, category, new_d)
            w.write(log)
    os.chdir(curdir)


def save_npy_data(dataset_name, root_dir, xtrain, xtest, ytrain, ytest):
    '''
        dataset_name: dataset name
        root_dir: dir for save npy data
        xtrain: array  [n1, window_size, modal_leng]
        xtest: array   [n2, window_size, modal_leng]
        ytrain: array  [n1,]
        ytest: array   [n2,]
    '''
    path = os.path.join(root_dir, dataset_name)
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(path + '/x_train.npy', xtrain)
    np.save(path + '/x_test.npy', xtest)
    np.save(path + '/y_train.npy', ytrain)
    np.save(path + '/y_test.npy', ytest)
    print('\n.npy data【xtrain，xtest，ytrain，ytest】has been saved in the directory【%s】\n' % (root_dir))
    build_npydataset_readme(root_dir)


widar_raw_all_dict = {
    "20181109-VS": {
        1: 'Push and Pull',
        2: 'Sweep',
        3: 'Clap',
        4: 'Slide',
        5: 'Draw Zigzag (Vertical)',
        6: 'Draw N (Vertical)'
    },
    "20181112-VS": {

    },
    "20181115-VS": {
        1: 'Push and Pull',
        2: 'Sweep',
        3: 'Clap',
        4: 'Draw Circle (Vertical)',
        5: 'Draw Zigzag (Vertical)',
        6: 'Draw N (Vertical)'
    },
    "20181116-VS": {},
    "20181117-VS": {
        1: 'Push and Pull',
        2: 'Sweep',
        3: 'Clap',
        4: 'Draw Circle (Vertical)',
        5: 'Draw Zigzag (Vertical)',
        6: 'Draw N (Vertical)'
    },
    "20181118-VS": {
        1: 'Push and Pull',
        2: 'Sweep',
        3: 'Clap',
        4: 'Draw Circle (Vertical)',
        5: 'Draw Zigzag (Vertical)',
        6: 'Draw N (Vertical)'
    },
    "20181121-VS": {
        1: 'Slide',
        2: 'Draw Circle (Horizontal)',
        3: 'Draw Zigzag (Horizontal)',
        4: 'Draw N (Horizontal)',
        5: 'Draw Triangle (Horizontal)',
        6: 'Draw Triangle (Horizontal)'
    },
    "20181127-VS": {
        1: 'Slide',
        2: 'Draw Circle (Horizontal)',
        3: 'Draw Zigzag (Horizontal)',
        4: 'Draw N (Horizontal)',
        5: 'Draw Triangle (Horizontal)',
        6: 'Draw Triangle (Horizontal)'
    },
    "20181128-VS": {
        1: 'Push and Pull',
        2: 'Sweep',
        3: 'Clap',
        4: 'Draw Circle (Horizontal)',
        5: 'Draw Zigzag (Horizontal)',
        6: 'Draw N (Horizontal)'
    },
    "20181130-VS": {
        1: 'Push and Pull',
        2: 'Sweep',
        3: 'Clap',
        4: 'Slide',
        5: 'Draw Circle (Horizontal)',
        6: 'Draw Zigzag (Horizontal)',
        7: 'Draw N (Horizontal)',
        8: 'Draw Triangle (Horizontal)',
        9: 'Draw Rectangle (Horizontal)',
    },
    "20181204-VS": {
        1: 'Push and Pull',
        2: 'Sweep',
        3: 'Clap',
        4: 'Slide',
        5: 'Draw Circle (Horizontal)',
        6: 'Draw Zigzag (Horizontal)',
        7: 'Draw N (Horizontal)',
        8: 'Draw Triangle (Horizontal)',
        9: 'Draw Rectangle (Horizontal)',
    },
    "20181205-VS": {

    },
    "20181208-VS": {

    },
    "20181209-VS": {

    },
    "20181211-VS": {
        1: 'Push and Pull',
        2: 'Sweep',
        3: 'Clap',
        4: 'Slide',
        5: 'Draw Circle (Horizontal)',
        6: 'Draw Zigzag (Horizontal)',
    },
}