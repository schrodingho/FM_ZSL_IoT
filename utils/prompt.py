import clip
import torch
import numpy as np
from collections import OrderedDict

def convert_to_token(xh):
    xh_id = clip.tokenize(xh).cpu().data.numpy()
    return xh_id


def text_prompt(meta_path, dataset='USC', clipbackbone='ViT-B/16', device='cpu', type='seen'):
    clipmodel, _ = clip.load(clipbackbone, device=device, jit=False)
    for paramclip in clipmodel.parameters():
        paramclip.requires_grad = False
    meta = open(meta_path, 'rb')

    actionlist = meta.readlines()
    meta.close()

    actionlist = np.array([a.decode('utf-8').split('\n')[0] for a in actionlist])
    numC = len(actionlist)
    actiontoken = np.array([convert_to_token(a) for a in actionlist])
    with torch.no_grad():
        actionembed = clipmodel.encode_text_light(torch.tensor(actiontoken).to(device))

    actiondict = OrderedDict((actionlist[i], actionembed[i].cpu().data.numpy()) for i in range(numC))
    actiontoken = OrderedDict((actionlist[i], actiontoken[i]) for i in range(numC))

    return actionlist, actiondict, actiontoken

