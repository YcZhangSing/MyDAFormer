# Copyright (c) OpenMMLab. All rights reserved.

# 原始的openmmlab的数据集转换代码 ，这个代码中，与DAFormer提供的cityscapes.py代码相比，并没有类别统计的功能
# （应该会影响后续的稀有类采样工作，但是可以在配置文件中将稀有类采样 关闭）
# 代码使用(py1.7-wzc)环境可以运行、

import argparse
import os.path as osp

from cityscapesscripts.preparation.json2labelImg import json2labelImg
from mmengine.utils import (mkdir_or_exist, scandir, track_parallel_progress,
                            track_progress)


def convert_json_to_label(json_file):
    label_file = json_file.replace('_polygons.json', '_labelTrainIds.png')
    json2labelImg(json_file, label_file, 'trainIds')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Cityscapes annotations to TrainIds')
    parser.add_argument('--cityscapes_path', help='cityscapes data path',default='/home/rui/WorkDir/yczhang/DAFormer/data/cityscapes')
    parser.add_argument('--gt-dir', default='gtFine', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cityscapes_path = args.cityscapes_path
    out_dir = args.out_dir if args.out_dir else cityscapes_path
    mkdir_or_exist(out_dir)

    gt_dir = osp.join(cityscapes_path, args.gt_dir)

    poly_files = []
    for poly in scandir(gt_dir, '_polygons.json', recursive=True):
        poly_file = osp.join(gt_dir, poly)
        poly_files.append(poly_file)
    if args.nproc > 1:
        track_parallel_progress(convert_json_to_label, poly_files, args.nproc)
    else:
        track_progress(convert_json_to_label, poly_files)

    split_names = ['train', 'val', 'test']

    for split in split_names:
        filenames = []
        for poly in scandir(
                osp.join(gt_dir, split), '_polygons.json', recursive=True):
            filenames.append(poly.replace('_gtFine_polygons.json', ''))
        with open(osp.join(out_dir, f'{split}.txt'), 'w') as f:
            f.writelines(f + '\n' for f in filenames)


if __name__ == '__main__':
    main()