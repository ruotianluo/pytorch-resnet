import os
os.environ["GLOG_minloglevel"] = "2"
import sys
import re
import caffe
import numpy as np
import skimage.io
from caffe.proto import caffe_pb2
from synset import *
import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
import resnet
from collections import OrderedDict

from torchvision import transforms as trn
trn_preprocess = trn.Compose([
        #trn.ToPILImage(),
        #trn.Scale(256),
        #trn.ToTensor(),
        #trn.Normalize([0.4829476, 0.4545211, 0.404167],[0.229, 0.224, 0.225])
        trn.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

class CaffeParamProvider():
    def __init__(self, caffe_net):
        self.caffe_net = caffe_net

    def conv_kernel(self, name):
        k = self.caffe_net.params[name][0].data
        if name == 'conv1':
            k = k[:,[2,1,0]]
            k *= 255.0
            k *= np.array([0.229, 0.224, 0.225])[np.newaxis,:,np.newaxis,np.newaxis]
        return k

    def bn_gamma(self, name):
        return self.caffe_net.params[name][0].data

    def bn_beta(self, name):
        return self.caffe_net.params[name][1].data

    def bn_mean(self, name):
        return self.caffe_net.params[name][0].data

    def bn_variance(self, name):
        return self.caffe_net.params[name][1].data

    def fc_weights(self, name):
        w = self.caffe_net.params[name][0].data
        #w = w.transpose((1, 0))
        return w

    def fc_biases(self, name):
        b = self.caffe_net.params[name][1].data
        return b


def preprocess(img):
    """Changes RGB [0,1] valued image to BGR [0,255] with mean subtracted."""
    mean_bgr = load_mean_bgr()
    print 'mean blue', np.mean(mean_bgr[:, :, 0])
    print 'mean green', np.mean(mean_bgr[:, :, 1])
    print 'mean red', np.mean(mean_bgr[:, :, 2])
    out = np.copy(img) * 255.0
    out = out[:, :, [2, 1, 0]]  # swap channel from RGB to BGR
    #out -= mean_bgr
    out -= mean_bgr.mean(0).mean(0)
    return out


def assert_almost_equal(caffe_tensor, th_tensor):
    t = th_tensor[0]
    c = caffe_tensor[0]

    #for i in range(0, t.shape[-1]):
    #    print "tf", i,  t[:,i]
    #    print "caffe", i,  c[:,i]

    if t.shape != c.shape:
        print "t.shape", t.shape
        print "c.shape", c.shape

    d = np.linalg.norm(t - c)
    print "d", d
    assert d < 500


# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path, size=224):
    img = skimage.io.imread(path)
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (size, size))
    return resized_img


def load_mean_bgr():
    """ bgr mean pixel value image, [0, 255]. [height, width, 3] """
    with open("data/ResNet_mean.binaryproto", mode='rb') as f:
        data = f.read()
    blob = caffe_pb2.BlobProto()
    blob.ParseFromString(data)

    mean_bgr = caffe.io.blobproto_to_array(blob)[0]
    assert mean_bgr.shape == (3, 224, 224)

    return mean_bgr.transpose((1, 2, 0))


def load_caffe(img_p, layers=50):
    caffe.set_mode_cpu()

    prototxt = "data/ResNet-%d-deploy.prototxt" % layers
    caffemodel = "data/ResNet-%d-model.caffemodel" % layers
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    net.blobs['data'].data[0] = img_p.transpose((2, 0, 1))
    assert net.blobs['data'].data[0].shape == (3, 224, 224)
    net.forward()

    caffe_prob = net.blobs['prob'].data[0]
    print_prob(caffe_prob)

    return net


# returns the top1 string
def print_prob(prob):
    #print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print "Top1: ", top1
    # Get top5 label
    top5 = [synset[pred[i]] for i in range(5)]
    print "Top5: ", top5
    return top1


def parse_pth_varnames(p, pth_varname, num_layers):
    if pth_varname == 'conv1.weight':
        return p.conv_kernel('conv1')

    elif pth_varname == 'bn1.weight':
        return p.bn_gamma('scale_conv1')

    elif pth_varname == 'bn1.bias':
        return p.bn_beta('scale_conv1')

    elif pth_varname == 'bn1.running_mean':
        return p.bn_mean('bn_conv1')

    elif pth_varname == 'bn1.running_var':
        return p.bn_variance('bn_conv1')

    elif pth_varname == 'fc.weight':
        return p.fc_weights('fc1000')

    elif pth_varname == 'fc.bias':
        return p.fc_biases('fc1000')

    # scale2/block1/shortcut/weights
    # scale3/block2/c/moving_mean
    # scale3/block6/c/moving_variance
    # scale4/block3/c/moving_mean
    # scale4/block8/a/beta
    # layer4.1.conv1.weight
    re1 = 'layer(\d+).(\d+).(downsample|conv1|bn1|conv2|bn2|conv3|bn3)'
    #re1 = 'scale(\d+)/block(\d+)/(shortcut|a|b|c|A|B)'
    m = re.search(re1, pth_varname)

    def letter(i):
        return chr(ord('a') + i - 1)

    scale_num = int(m.group(1)) + 1

    block_num = int(m.group(2)) + 1
    if scale_num == 2:
        # scale 2 always uses block letters
        block_str = letter(block_num)
    elif scale_num == 3 or scale_num == 4:
        # scale 3 uses block letters for l=50 and numbered blocks for l=101, l=151
        # scale 4 uses block letters for l=50 and numbered blocks for l=101, l=151
        if num_layers == 50:
            block_str = letter(block_num)
        else:
            if block_num == 1:
                block_str = 'a'
            else:
                block_str = 'b%d' % (block_num - 1)
    elif scale_num == 5:
        # scale 5 always block letters
        block_str = letter(block_num)
    else:
        raise ValueError("unexpected scale_num %d" % scale_num)

    branch = m.group(3)
    if branch == "downsample":
        branch_num = 1
        conv_letter = ''
    else:
        branch_num = 2
        conv_letter = letter(int(branch[-1]))

    x = (scale_num, block_str, branch_num, conv_letter)
    #print x
    #print pth_varname, '\t', 

    if ('weight' in pth_varname and 'conv' in pth_varname) or 'downsample.0.weight' in pth_varname:
        #print 'res%d%s_branch%d%s' % x
        return p.conv_kernel('res%d%s_branch%d%s' % x)

    if ('weight' in pth_varname and 'bn' in pth_varname) or 'downsample.1.weight' in pth_varname:
        #print 'scale%d%s_branch%d%s' % x
        return p.bn_gamma('scale%d%s_branch%d%s' % x)

    if ('bias' in pth_varname and 'bn' in pth_varname) or 'downsample.1.bias' in pth_varname:
        #print 'scale%d%s_branch%d%s' % x
        return p.bn_beta('scale%d%s_branch%d%s' % x)

    if ('running_mean' in pth_varname and 'bn' in pth_varname) or 'downsample.1.running_mean' in pth_varname:
        #print 'bn%d%s_branch%d%s' % x
        return p.bn_mean('bn%d%s_branch%d%s' % x)

    if ('running_var' in pth_varname and 'bn' in pth_varname) or 'downsample.1.running_var' in pth_varname:
        #print 'bn%d%s_branch%d%s' % x
        return p.bn_variance('bn%d%s_branch%d%s' % x)

    raise ValueError('unhandled var ' + pth_varname)


def checkpoint_fn(layers):
    return 'resnet%d.pth' % layers

def convert(img, img_p, layers):
    caffe_model = load_caffe(img_p, layers)

    #for i, n in enumerate(caffe_model.params):
    #    print n

    param_provider = CaffeParamProvider(caffe_model)

    if layers == 50:
        num_blocks = [3, 4, 6, 3]
    elif layers == 101:
        num_blocks = [3, 4, 23, 3]
    elif layers == 152:
        num_blocks = [3, 8, 36, 3]

    model = getattr(resnet, 'resnet'+str(layers))()
    model.eval()

    #from copy import deepcopy
    #new_state_dict = deepcopy(model.state_dict())
    new_state_dict = OrderedDict()

    for var_name in model.state_dict():
        #print var.op.name
        data = parse_pth_varnames(param_provider, var_name, layers)
        #print "caffe data shape", data.shape
        #print "tf shape", var.get_shape()
        new_state_dict[var_name] = torch.from_numpy(data).float()
    model.load_state_dict(new_state_dict)

    o = []
    def hook(module, input, output):
        #print module
        o.append(input[0].data.numpy())

    model.maxpool.register_forward_hook(hook)
    model.layer1._modules['0'].conv1.register_forward_hook(hook)
    model.layer1._modules['1'].conv1.register_forward_hook(hook)
    model.layer1._modules['2'].conv1.register_forward_hook(hook)
    model.layer2._modules['0'].conv1.register_forward_hook(hook)

    # model.layer2._modules['0'].conv2.register_forward_hook(hook)
    # model.layer2._modules['0'].conv3.register_forward_hook(hook)

    model.layer2._modules['1'].conv1.register_forward_hook(hook)
    model.avgpool.register_forward_hook(hook)
    model.fc.register_forward_hook(hook)
    #model.fc.register_forward_hook(hook)

    #output_prob = model(Variable(torch.from_numpy(img_p[np.newaxis, :].transpose([0,3,1,2])).float(), volatile=True))
    I = torch.from_numpy(img.transpose([2,0,1])).float()
    output_prob = model(Variable(trn_preprocess(I).unsqueeze(0), volatile=True))

    assert_almost_equal(caffe_model.blobs['conv1'].data, o[0])
    assert_almost_equal(caffe_model.blobs['pool1'].data, o[1])
    assert_almost_equal(caffe_model.blobs['res2a'].data, o[2])
    assert_almost_equal(caffe_model.blobs['res2b'].data, o[3])
    assert_almost_equal(caffe_model.blobs['res2c'].data, o[4])
    # assert_almost_equal(caffe_model.blobs['res3a_branch2a'].data, o[5])
    # assert_almost_equal(caffe_model.blobs['res3a_branch2b'].data, o[6])
    assert_almost_equal(caffe_model.blobs['res3a'].data, o[5])
    assert_almost_equal(caffe_model.blobs['res5c'].data, o[6])
    assert_almost_equal(caffe_model.blobs['pool5'].data[:,:,0,0], o[7])
    
    #print_prob(o[8][0])
    th_prob = F.softmax(output_prob[0]).data.numpy()
    print_prob(th_prob)

    prob_dist = np.linalg.norm(caffe_model.blobs['prob'].data - th_prob)
    print 'prob_dist ', prob_dist
    assert prob_dist < 0.2  # XXX can this be tightened?

    # Save the model
    torch.save(model.state_dict(), checkpoint_fn(layers))


def main():
    img = load_image("data/cat.jpg")
    print img
    img_p = preprocess(img)

    for layers in [50, 101, 152]:
        print "CONVERT", layers
        convert(img, img_p, layers)


if __name__ == '__main__':
    main()