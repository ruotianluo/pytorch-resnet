""""""""""""""""""""""""""""
This conversion is partly borrowed from Detectron.pytorch
""""""""""""""""""""""""""""

import os
os.environ["GLOG_minloglevel"] = "2"
import sys
import re
# import caffe
import numpy as np
import skimage.io
# from caffe.proto import caffe_pb2
from synset import *
import torch
import torchvision.models as models
import torch.nn.functional as F
import resnet
from collections import OrderedDict

import cPickle as pickle

import argparse
parser = argparse.ArgumentParser(description='Convert group norm checkpoints')
parser.add_argument('--layers', default='50', type=str,
                    help='50 or 101')
parser.add_argument('--mode', default='pth', type=str,
                    help='pth or caffe')

args = parser.parse_args()

# def resnet_weights_name_pattern():
#     pattern = re.compile(r"conv1_w|conv1_gn_[sb]|res_conv1_.+|res\d+_\d+_.+")
#     return pattern

if not os.path.exists('data/R-%s-GN.pkl' %args.layers):
    if args.layers == '50':
        os.system('cd data;wget https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/47261647/R-50-GN.pkl')
    elif args.layers == '101':
        os.system('cd data;wget https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/47592356/R-101-GN.pkl')

with open('data/R-%s-GN.pkl' %args.layers, 'rb') as fp:
    src_blobs = pickle.load(fp)
    if 'blobs' in src_blobs:
        src_blobs = src_blobs['blobs']
    pretrianed_state_dict = src_blobs

import resnet

model = getattr(resnet, 'resnet%s_gn' %args.layers)()
model.eval()

model_state_dict = model.state_dict()

def detectron_weight_mapping(self):
    mapping_to_detectron = {
        'conv1.weight': 'conv1_w',
        'bn1.weight': 'conv1_gn_s',
        'bn1.bias': 'conv1_gn_b'
    }

    for res_id in range(1, 5):
        stage_name = 'layer%d' % res_id
        mapping = residual_stage_detectron_mapping(
            getattr(self, stage_name), res_id)
        mapping_to_detectron.update(mapping)

    return mapping_to_detectron

def residual_stage_detectron_mapping(module_ref, res_id):
    """Construct weight mapping relation for a residual stage with `num_blocks` of
    residual blocks given the stage id: `res_id`
    """
    pth_norm_suffix = '_bn'
    norm_suffix = '_gn'
    mapping_to_detectron = {}
    for blk_id in range(len(module_ref)):
        detectron_prefix = 'res%d_%d' % (res_id+1, blk_id)
        my_prefix = 'layer%s.%d' % (res_id, blk_id)

        # residual branch (if downsample is not None)
        if getattr(module_ref[blk_id], 'downsample'):
            dtt_bp = detectron_prefix + '_branch1'  # short for "detectron_branch_prefix"
            mapping_to_detectron[my_prefix
                                 + '.downsample.0.weight'] = dtt_bp + '_w'
            mapping_to_detectron[my_prefix
                                 + '.downsample.1.weight'] = dtt_bp + norm_suffix + '_s'
            mapping_to_detectron[my_prefix
                                 + '.downsample.1.bias'] = dtt_bp + norm_suffix + '_b'

        # conv branch
        for i, c in zip([1, 2, 3], ['a', 'b', 'c']):
            dtt_bp = detectron_prefix + '_branch2' + c
            mapping_to_detectron[my_prefix
                                 + '.conv%d.weight' % i] = dtt_bp + '_w'
            mapping_to_detectron[my_prefix
                                 + '.' + pth_norm_suffix[1:] + '%d.weight' % i] = dtt_bp + norm_suffix + '_s'
            mapping_to_detectron[my_prefix
                                 + '.' + pth_norm_suffix[1:] + '%d.bias' % i] = dtt_bp + norm_suffix + '_b'

    return mapping_to_detectron

name_mapping = detectron_weight_mapping(model)
name_mapping.update({
    'fc.weight': 'pred_w',
    'fc.bias': 'pred_b'
})

assert set(model_state_dict.keys()) == set(name_mapping.keys())
assert set(pretrianed_state_dict.keys()) == set(name_mapping.values())

# pattern = resnet_weights_name_pattern()
for k, v in name_mapping.items():
    if isinstance(v, str):  # maybe a str, None or True
        if True: #pattern.match(v):
            pretrianed_key = k.split('.', 1)[-1]
            assert(model_state_dict[k].shape == torch.Tensor(pretrianed_state_dict[v]).shape)
            model_state_dict[k].copy_(torch.Tensor(pretrianed_state_dict[v]))
        if k == 'conv1.weight' and args.mode == 'pth':
            tmp = model_state_dict[k]
            tmp = tmp[:,[2,1,0]].numpy()
            tmp *= 255.0
            tmp *= np.array([0.229, 0.224, 0.225])[np.newaxis,:,np.newaxis,np.newaxis]
            model_state_dict[k].copy_(torch.from_numpy(tmp))
torch.save(model_state_dict, 'resnet_gn%s-%s.pth'%(args.layers, args.mode))
print('Converted state dict saved to %s' %('resnet_gn%s-%s.pth'%(args.layers, args.mode)))