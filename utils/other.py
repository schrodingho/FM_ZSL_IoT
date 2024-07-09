import os
import torch
from glob import glob
import shutil

def set_learning_rate(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr


def readtxt(metapath, datapath):
    vidDir, vidLabel = [], []
    f = open(metapath, 'rb')
    path = f.readlines()
    f.close()
    for p in path:
        psplit = p.decode('utf-8').strip('\n').split(',')
        vidDir += [os.path.join(datapath, psplit[0])]
        vidLabel += [[int(psplit[1]), psplit[2], int(psplit[3])]]
    return vidDir, vidLabel


def save_checkpoint(state, is_best=0, gap=1, filename='checkpoint.pth.tar', keep_all=False):
    torch.save(state, filename)
    last_epoch_path = os.path.join(os.path.dirname(filename), 'checkpoint_iter%s.pth.tar' % str(state['iteration']-gap))
    if not keep_all:
        try:
            os.remove(last_epoch_path)
        except:
            pass

    if is_best:
        past_best = glob(os.path.join(os.path.dirname(filename), 'model_best_*.pth.tar'))
        past_best = sorted(past_best, key=lambda x: int(''.join(filter(str.isdigit, x))))
        if len(past_best) >= 3:
            try: os.remove(past_best[0])
            except: pass
        torch.save(state, os.path.join(os.path.dirname(filename), 'model_best_iter%s.pth.tar' % str(state['iteration'])))

def save_best_checkpoint(state, is_best=0, gap=1, filename='checkpoint.pth.tar', keep_all=False):
    if is_best:
        past_best = glob(os.path.join(os.path.dirname(filename), 'model_best_*.pth.tar'))
        past_best = sorted(past_best, key=lambda x: int(''.join(filter(str.isdigit, x))))
        if len(past_best) >= 3:
            try: os.remove(past_best[0])
            except: pass
        torch.save(state, os.path.join(os.path.dirname(filename), 'model_best_iter%s.pth.tar' % str(state['iteration'])))

def save_unseen_best(state, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(os.path.dirname(filename), 'model_best.pth.tar'))

def save_best_checkpoint_epoch(state, is_best=0, gap=1, filename='checkpoint.pth.tar', keep_all=False):
    if is_best:
        past_best = glob(os.path.join(os.path.dirname(filename), 'model_best_*.pth.tar'))
        past_best = sorted(past_best, key=lambda x: int(''.join(filter(str.isdigit, x))))
        if len(past_best) >= 2:
            try: os.remove(past_best[0])
            except: pass
        torch.save(state, os.path.join(os.path.dirname(filename), 'model_best_epoch%s.pth.tar' % str(state['epoch'])))

def delete_redundant_epoch_dirs(directory):
    files_and_folders = os.listdir(directory)
    epoch_folders = [folder for folder in files_and_folders if
                     os.path.isdir(os.path.join(directory, folder)) and folder.startswith('epoch')]
    sorted_folders = sorted(epoch_folders, key=lambda x: os.path.getmtime(os.path.join(directory, x)))

    for folder in sorted_folders[:-1]:
        folder_path = os.path.join(directory, folder)
        try:
            shutil.rmtree(folder_path)  # Use shutil.rmtree to delete non-empty directories recursively
        except OSError as e:
            print(f"Error: {e.strerror} - {folder_path}")


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class FastDataLoader(torch.utils.data.dataloader.DataLoader):
    '''for reusing cpu workers, to save time'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        # self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)
