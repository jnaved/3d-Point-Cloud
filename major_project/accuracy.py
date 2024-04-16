import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from easydict import EasyDict as edict
import torch
import numpy as np

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("jnaved"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from data_process.kitti_dataloader import create_test_dataloader
from models.model_utils import create_model
from utils.misc import make_folder, time_synchronized
from utils.evaluation_utils import decode, post_processing
from utils.torch_utils import sigmoid
from calculate_accuracy import get_corners, calculate_iou


def parse_eval_configs():
    parser = argparse.ArgumentParser(description='Accuracy config')
    parser.add_argument('--saved_fn', type=str, default='fpn_resnet_18', metavar='FN',
                        help='The name using for saving logs, models,...')
    parser.add_argument('-a', '--arch', type=str, default='fpn_resnet_18', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--pretrained_path', type=str,
                        default='../checkpoints/fpn_resnet_18/Model_fpn_resnet_18_epoch_200.pth', metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--actual_path', type=str,
                        default='../checkpoints/fpn_resnet_18/fpn_resnet_18_epoch_300.pth', metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--K', type=int, default=50,
                        help='the number of top K')
    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=0, type=int,
                        help='GPU index to use.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 4)')
    parser.add_argument('--peak_thresh', type=float, default=0.2)

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True
    configs.distributed = False  # For testing on 1 GPU only

    configs.input_size = (608, 608)
    configs.hm_size = (152, 152)
    configs.down_ratio = 4
    configs.max_objects = 50

    configs.imagenet_pretrained = False
    configs.head_conv = 64
    configs.num_classes = 3
    configs.num_center_offset = 2
    configs.num_z = 1
    configs.num_dim = 3
    configs.num_direction = 2  # sin, cos

    configs.heads = {
        'hm_cen': configs.num_classes,
        'cen_offset': configs.num_center_offset,
        'direction': configs.num_direction,
        'z_coor': configs.num_z,
        'dim': configs.num_dim
    }
    configs.num_input_features = 4

    ####################################################################
    ##############Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    configs.root_dir = '../'
    configs.dataset_dir = os.path.join(configs.root_dir, 'dataset', 'kitti')

    return configs


if __name__ == '__main__':
    configs = parse_eval_configs()

    model1 = create_model(configs)
    model2 = create_model(configs)
    print('\n\n' + '-*=' * 30 + '\n\n')
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    assert os.path.isfile(configs.actual_path), "No file at {}".format(configs.actual_path)
    model1.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
    model2.load_state_dict(torch.load(configs.actual_path, map_location='cpu'))

    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model1 = model1.to(device=configs.device)
    model2 = model2.to(device=configs.device)

    out_cap = None

    model1.eval()
    model2.eval()

    eval_dataloader = create_test_dataloader(configs)
    with torch.no_grad():
        avg_iou = 0
        for batch_idx, batch_data in enumerate(eval_dataloader):
            metadatas, bev_maps, img_rgbs = batch_data
            input_bev_maps = bev_maps.to(configs.device, non_blocking=True).float()
            t1 = time_synchronized()
            outputs1 = model1(input_bev_maps)
            outputs2 = model2(input_bev_maps)
            outputs1['hm_cen'] = sigmoid(outputs1['hm_cen'])
            outputs1['cen_offset'] = sigmoid(outputs1['cen_offset'])
            outputs2['hm_cen'] = sigmoid(outputs2['hm_cen'])
            outputs2['cen_offset'] = sigmoid(outputs2['cen_offset'])
            # detections size (batch_size, K, 10)
            detections1 = decode(outputs1['hm_cen'], outputs1['cen_offset'], outputs1['direction'], outputs1['z_coor'],
                                 outputs1['dim'], K=configs.K)
            detections2 = decode(outputs2['hm_cen'], outputs2['cen_offset'], outputs2['direction'], outputs2['z_coor'],
                                 outputs2['dim'], K=configs.K)
            detections1 = detections1.cpu().numpy().astype(np.float32)
            detections2 = detections2.cpu().numpy().astype(np.float32)

            detections1 = post_processing(detections1, configs.num_classes, configs.down_ratio, configs.peak_thresh)
            detections2 = post_processing(detections2, configs.num_classes, configs.down_ratio, configs.peak_thresh)
            t2 = time_synchronized()

            detections1 = detections1[0]  # only first batch
            detections2 = detections2[0]

            total_iou = 0
            count = 0
            for j in range(configs.num_classes):
                n = len(detections1[j])
                m = len(detections2[j])
                iou = 0
                for i in range(min(n, m)):
                    det1 = detections1[j][i]
                    det2 = detections2[j][i]
                    # (scores-0:1, x-1:2, y-2:3, z-3:4, dim-4:7, yaw-7:8)
                    pred_x, pred_y, pred_w, pred_l, pred_yaw = det1[1], det1[2], det1[5], det1[6], det1[7]
                    gt_x, gt_y, gt_w, gt_l, gt_yaw = det2[1], det2[2], det2[5], det2[6], det2[7]

                    pred_corners = get_corners(pred_x, pred_y, pred_w, pred_l, pred_yaw)
                    gt_corners = get_corners(gt_x, gt_y, gt_w, gt_l, gt_yaw)

                    iou += calculate_iou(gt_corners, pred_corners)
                    count += 1  
                total_iou += iou / count if count != 0 else 0
            avg_iou += total_iou/count if count != 0 else 0
        print(avg_iou * 100/batch_idx)
