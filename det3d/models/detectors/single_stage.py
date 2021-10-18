import torch.nn as nn

from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from ..utils.finetune_utils import FrozenBatchNorm2d
from det3d.torchie.trainer import load_checkpoint


@DETECTORS.register_module
class SingleStageDetector(BaseDetector):
    def __init__(
        self,
        reader,
        backbone,
        encoder2d = None,
        neck=None,
        cfe = None,
        decoder2d = None,
        bbox_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(SingleStageDetector, self).__init__()
        self.reader = builder.build_reader(reader)

        if backbone is not None:
            self.backbone = builder.build_backbone(backbone)
            
        if neck is not None:
            self.neck = builder.build_neck(neck)


        if encoder2d is not None:
            self.encoder2d = builder.build_encoder(encoder2d)

        if cfe is not None:
            self.cfe = builder.build_cfe(cfe)

        if decoder2d is not None:
            self.decoder2d = builder.build_decoder(decoder2d)

            
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        if pretrained is None:
            return 
        try:
            load_checkpoint(self, pretrained, strict=False)
            print("init weight from {}".format(pretrained))
        except:
            print("no pretrained model at {}".format(pretrained))
            
    def extract_feat(self, data):
        input_features = self.reader(data)
        x = self.backbone(input_features)
        if self.with_neck:
            x = self.neck(x)
        return x

    def aug_test(self, example, rescale=False):
        raise NotImplementedError

    def forward(self, example, return_loss=True, **kwargs):
        pass

    def predict(self, example, preds_dicts):
        pass

    def freeze(self):
        for name, p in self.named_parameters():
            p.requires_grad = False
            # print("Freezing parameter")
            # print(name)

        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self