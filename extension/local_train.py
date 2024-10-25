import os
import torch
import torch.nn as nn
import time
import logging
from utils.other import save_local_best

def train_supervised_model(config, dataloader, model, optimizer, lr_scheduler, device):
    loss_func = nn.CrossEntropyLoss()
    timestart = time.time()
    iteration = config["args"]["start_iter"]
    trnloader, valloader, val_mix_loader = dataloader
    iteration_epoch = len(trnloader)
    dataset = config["dataset_args"]["dataset"]

    model.train()
    best_acc = 0
    epoch = 0
    epochs = config["args"]["epochs"]
    while epoch < epochs:
        for idx, sample in enumerate(trnloader):
            vids, _, y_true, _ = sample
            targets_real = y_true.to(device)

            pred = model(vids.to(device))

            loss = loss_func(pred, targets_real)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            if iteration % 100 == 0:
                print('dataset: {},'.format(dataset),
                      'epoch[{}][{}/{}]'.format(iteration // iteration_epoch, idx, iteration_epoch),
                      'epoch[{}], '.format(epoch),
                      'time {:.01f}s,'.format(time.time() - timestart),
                      'loss {:.03f},'.format(loss.detach().cpu().numpy()),
                      )
                timestart = time.time()
            iteration += 1
        if epoch >= config["args"]["val_epoch"]:
            acc = val_supervised_model(config, model, valloader, device)
            if acc > best_acc:
                best_acc = acc
                print(f"Epoch {epoch} best_seen_acc: {best_acc}")
                logging.info(f"Epoch {epoch} best_seen_acc: {best_acc}")
                state_dict = model.state_dict()
                save_dict = {
                    'state_dict': state_dict,
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch
                }
                print("Saving best checkpoint")
                config["best_model_path"] = os.path.join(config["model_path"], 'local_model_best.pth.tar')
                save_local_best(save_dict, path=config["best_model_path"])
        model.train()

        epoch += 1


def val_supervised_model(config, model, dataloader, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for idx, sample in enumerate(dataloader):
            vids, _, y_true, _ = sample

            targets_real = y_true.to(device)
            pred = model(vids.to(device))

            _, predicted = torch.max(pred, 1)
            total += targets_real.size(0)
            correct += (predicted == targets_real).sum().item()

        acc = correct / total
        print(f"{config['dataset_args']['dataset']} acc: {acc}")
        logging.info(f"{config['dataset_args']['dataset']} acc: {acc}")
    return acc