# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Support UDA models
# 这段代码定义了一个用于构建模型的工具函数 `build_segmentor`，并且定义了一个 `UDA` 类，它是 `MODELS` 注册表的一个别名。

# 这段代码的功能包括：

# 1. 定义了一个 `MODELS` 注册表，用于注册模型结构。
# 2. 定义了一个 `ATTENTION` 注册表，用于注册注意力机制。
# 3. 定义了一系列函数用于构建模型的不同部分（backbone、neck、head、loss 等）。
# 4. 定义了 `build_train_model` 和 `build_segmentor` 函数，用于根据配置文件构建模型。

# 总的来说，这段代码提供了一种统一的接口来构建模型，并提供了一些默认参数，使得构建模型变得更加方便。
####
import warnings

from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.cnn.bricks.registry import ATTENTION as MMCV_ATTENTION
from mmcv.utils import Registry

MODELS = Registry('models', parent=MMCV_MODELS)
ATTENTION = Registry('attention', parent=MMCV_ATTENTION)

BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
SEGMENTORS = MODELS
UDA = MODELS


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_train_model(cfg, train_cfg=None, test_cfg=None):
    """Build model."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.model.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.model.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    if 'uda' in cfg:
        cfg.uda['model'] = cfg.model
        cfg.uda['max_iters'] = cfg.runner.max_iters
        return UDA.build(
            cfg.uda, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
    else:
        return SEGMENTORS.build(
            cfg.model,
            default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_segmentor(cfg, train_cfg=None, test_cfg=None):
    """Build segmentor."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return SEGMENTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
