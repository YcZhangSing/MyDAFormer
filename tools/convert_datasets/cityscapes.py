# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add class stats computation

import argparse
import json
import os.path as osp

import mmcv
import numpy as np
from cityscapesscripts.preparation.json2labelImg import json2labelImg
from PIL import Image

## 使用新版的mmcv的scandir不会出现文件转换的问题
## 同样的使用较高版本的mmcv  py1.7-wzc环境可以运行
from mmengine.utils import (mkdir_or_exist, scandir, track_parallel_progress,
                            track_progress)
##########################

def convert_json_to_label(json_file):
    label_file = json_file.replace('_polygons.json', '_labelTrainIds.png')
    json2labelImg(json_file, label_file, 'trainIds')

    # 在训练集中统计每个类别的样本数（cityscapes共有19类样本）
    if 'train/' in json_file:
        pil_label = Image.open(label_file)
        label = np.asarray(pil_label)
        sample_class_stats = {}
        for c in range(19):
            n = int(np.sum(label == c))
            if n > 0:
                sample_class_stats[int(c)] = n
        sample_class_stats['file'] = label_file
        return sample_class_stats
    else:
        return None


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


def save_class_stats(out_dir, sample_class_stats):
    sample_class_stats = [e for e in sample_class_stats if e is not None]
    with open(osp.join(out_dir, 'sample_class_stats.json'), 'w') as of:
        # 保存每个类别的样本数
        json.dump(sample_class_stats, of, indent=2)

    sample_class_stats_dict = {}
    for stats in sample_class_stats:
        f = stats.pop('file')
        sample_class_stats_dict[f] = stats
    # 保存每个文件中每个类别的样本数
    with open(osp.join(out_dir, 'sample_class_stats_dict.json'), 'w') as of:
        json.dump(sample_class_stats_dict, of, indent=2)

    samples_with_class = {}
    for file, stats in sample_class_stats_dict.items():
        for c, n in stats.items():
            # 统计每个类别的样本数
            if c not in samples_with_class:
                samples_with_class[c] = [(file, n)]
            # 统计每个类别的样本数
            else:
                samples_with_class[c].append((file, n))
    # 保存每个类别的样本数
    with open(osp.join(out_dir, 'samples_with_class.json'), 'w') as of:
        json.dump(samples_with_class, of, indent=2)


def main():
    args = parse_args()
    cityscapes_path = args.cityscapes_path
    out_dir = args.out_dir if args.out_dir else cityscapes_path
    mmcv.mkdir_or_exist(out_dir)

    gt_dir = osp.join(cityscapes_path, args.gt_dir)

    poly_files = []
    for poly in scandir(gt_dir, '_polygons.json', recursive=True):
        poly_file = osp.join(gt_dir, poly)
        poly_files.append(poly_file)


    only_postprocessing = False
    # only_postprocessing = True
    
    if not only_postprocessing:
        if args.nproc > 1:
            sample_class_stats = track_parallel_progress(
                convert_json_to_label, poly_files, args.nproc)
        else:
            sample_class_stats = track_progress(convert_json_to_label,
                                                     poly_files)
    else:
        with open(osp.join(out_dir, 'sample_class_stats.json'), 'r') as of:
            sample_class_stats = json.load(of)

    save_class_stats(out_dir, sample_class_stats)

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
