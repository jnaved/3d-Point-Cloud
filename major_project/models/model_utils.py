import os
import sys

import torch

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("jnaved"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from models import resnet, fpn_resnet


def create_model(configs):
    """Create model based on architecture name"""
    try:
        arch_parts = configs.arch.split('_')
        num_layers = int(arch_parts[-1])
    except:
        raise ValueError
    if 'fpn_resnet' in configs.arch:
        print('using ResNet architecture with feature pyramid')
        model = fpn_resnet.get_pose_net(num_layers=num_layers, heads=configs.heads, head_conv=configs.head_conv,
                                        imagenet_pretrained=configs.imagenet_pretrained)
    elif 'resnet' in configs.arch:
        print('using ResNet architecture')
        model = resnet.get_pose_net(num_layers=num_layers, heads=configs.heads, head_conv=configs.head_conv,
                                    imagenet_pretrained=configs.imagenet_pretrained)
    else:
        assert False, 'Undefined model backbone'

    return model


def get_num_parameters(model):
    """Count number of trained parameters of the model"""
    if hasattr(model, 'module'):
        num_parameters = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
    else:
        num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return num_parameters


if __name__ == '__main__':
    import argparse
    from easydict import EasyDict as edict

    parser = argparse.ArgumentParser(description='Major Project')
    parser.add_argument('-a', '--arch', type=str, default='resnet_18', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--head_conv', type=int, default=-1,
                        help='conv layer channels for output head'
                             '0 for no conv layer'
                             '-1 for default setting: '
                             '64 for resnets and 256 for dla.')

    configs = edict(vars(parser.parse_args()))
    if configs.head_conv == -1:  # init default head_conv
        configs.head_conv = 256 if 'dla' in configs.arch else 64

    configs.num_classes = 3
    configs.num_vertexes = 8
    configs.num_center_offset = 2
    configs.num_vertexes_offset = 2
    configs.num_dimension = 3
    configs.num_rot = 8
    configs.num_depth = 1
    configs.num_wh = 2
    configs.heads = {
        'hm_mc': configs.num_classes,
        'hm_ver': configs.num_vertexes,
        'vercoor': configs.num_vertexes * 2,
        'cenoff': configs.num_center_offset,
        'veroff': configs.num_vertexes_offset,
        'dim': configs.num_dimension,
        'rot': configs.num_rot,
        'depth': configs.num_depth,
        'wh': configs.num_wh
    }

    configs.device = torch.device('cuda:1')

    model = create_model(configs).to(device=configs.device)
    sample_input = torch.randn((1, 3, 224, 224)).to(device=configs.device)
    output = model(sample_input)
    for hm_name, hm_out in output.items():
        print('hm_name: {}, hm_out size: {}'.format(hm_name, hm_out.size()))

    print('number of parameters: {}'.format(get_num_parameters(model)))
