'''
Reproducing paper:
An Extremely Efficient Convolutional Neural Network for Mobile Devices, Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, Jian Sun
'''
import mxnet as mx

from ..operators.op_channelshuffle import *

def shufflenet_unit(data, num_filter, stride, dim_match, name, num_group, bn_mom=0.9, workspace=512, memonger=False):
    """Return Shufflenet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    stride : tupe
        Stride used in convolution
    dim_match : Boolen
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    num_group: int
        Number of groups
    workspace : int
        Workspace used in convolution operator
    """

    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
    conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.25), num_group=num_group, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    no_bias=True, workspace=workspace, name=name + '_conv1')

    # shuffle channels
    sc1 = mx.symbol.Custom(data=conv1, name='shufflechannel', op_type='shufflechannel', num_group=num_group)
    bn2 = mx.sym.BatchNorm(data=sc1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
    act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')

    # Depthwise Conv, method 1: using group strategy

    # conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter*0.25), num_group=int(num_filter*0.25), kernel=(3,3), stride=stride, pad=(1,1),
    #                                 no_bias=True, workspace=workspace, name=name + '_conv2')

    # Depthwise Conv, method 2: using channel slice
    sc1_channels = mx.sym.split(data=act2, axis=1, num_outputs=int(num_filter*0.25))
    dw_outs = [mx.sym.Convolution(data=sc1_channels[i], num_filter=1, pad=(1, 1), kernel=(3, 3), stride=stride, 
             no_bias=True, workspace=workspace, name=name+'_conv2_depthwise_kernel'+str(i)) for i in range(int(num_filter*0.25))]

    conv2 = mx.sym.Concat(*dw_outs)   
    # 
    bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
    act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
    conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, num_group=num_group, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                workspace=workspace, name=name + '_conv3')
    if dim_match:
        shortcut = data
    else:
        shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                        workspace=workspace, name=name+'_sc')
    if memonger:
        shortcut._set_attr(mirror_stage='True')
    return conv3 + shortcut


def shufflenet(units, num_stage, filter_list, num_class, num_group, bn_mom=0.9, workspace=512, memonger=False):
    """Return Shufflenet symbol

    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stage : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_class : int
        Ouput size of symbol
    num_group : int
        Number of groups
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator

    In paper, the units is [4, 8, 4]
    when num_group=3, filter_list is [24, 240, 480, 960]
    """
    
    num_unit = len(units)
    assert(num_unit == num_stage)
    data = mx.sym.Variable(name='data')
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')

    body = mx.sym.Convolution(data=data, num_filter=filter_list[0], num_group=num_group, kernel=(3, 3), stride=(2,2), pad=(1, 1),
                                no_bias=True, name="conv0", workspace=workspace)
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
    body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
    body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')

    for i in range(num_stage):
        body = shufflenet_unit(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), num_group=num_group, workspace=workspace,
                             memonger=memonger)
        for j in range(units[i]-1):
            body = shufflenet_unit(body, filter_list[i+1], (1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                             num_group=num_group, workspace=workspace, memonger=memonger)

    bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
    # Although kernel is not used here when global_pool=True, we should put one
    pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    flat = mx.symbol.Flatten(data=pool1)
    fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=num_class, name='fc1')
    return mx.symbol.SoftmaxOutput(data=fc1, name='softmax')
