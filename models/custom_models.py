import torch
import torch.nn as nn
import torch.nn.functional as F
import timm_latest as timm
import segmentation_models_pytorch as smp
from kuma_utils.torch.modules import CBAM2d, SpatialAttention
from kuma_utils.torch.utils import freeze_module


class CustomResNet200D(nn.Module):
    def __init__(self, model_name='resnet200d_320', 
                 pretrained=False, num_classes=11, attention=None, return_mask=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        if attention is not None:
            if attention in ['CBAM', 'cbam']:
                self.attention = CBAM2d(in_planes=n_features, return_mask=True)
        else:
            self.attention = None
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, num_classes)
        self.return_mask = return_mask

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        if self.attention is not None:
            attn_mask, features = self.attention(features)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(pooled_features)
        if self.return_mask:
            return attn_mask, pooled_features, output
        else:
            return features, pooled_features, output


class CustomEfficientNet(nn.Module):
    def __init__(self, model_name="se_resnext50_32x4d", pretrained=False,
                 num_classes=11, attention=None, **kwargs):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            **kwargs)
        n_features = self.model.classifier.in_features
        self.model.global_pool = nn.Identity()
        self.model.classifier = nn.Identity()
        if attention is not None:
            if attention in ['CBAM', 'cbam']:
                self.attention = CBAM2d(in_planes=n_features)
        else:
            self.attention = None
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, num_classes)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        if self.attention is not None:
            features = (1+self.attention(features)) * features
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(pooled_features)
        return features, pooled_features, output


class CustomNFNet(nn.Module):
    def __init__(self, model_name='dm_nfnet_f0', pretrained=False, num_classes=11):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.head.fc.in_features
        self.model.head.global_pool = nn.Identity()
        self.model.head.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, num_classes)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(pooled_features)
        return output


'''
Segmenation and classification models
'''
class SegmentationAndClassification(nn.Module):

    def __init__(self,
                 segmentation_model='se_resnext50_32x4d',
                 segmentation_classes=5,
                 segmentation_params={},
                 classification_model='resnet18',
                 classification_classes=11,
                 classification_params={},
                 in_channels=3,
                 pretrained=False,
                 return_mask=False, return_feature=False,
                 freeze_segmentation=False,
                 concat_original=False):

        super().__init__()
        self.return_mask = return_mask
        self.return_feature = return_feature
        self.freeze_segmentation = freeze_segmentation
        self.concat = concat_original

        self.segmentation_model = smp.Unet(
            encoder_name=segmentation_model,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=segmentation_classes,
            **segmentation_params
        )
        self.classification_model = timm.create_model(
            classification_model,
            pretrained=pretrained,
            in_chans=segmentation_classes+in_channels if self.concat else segmentation_classes,
            num_classes=classification_classes,
            **classification_params
        )
        if self.freeze_segmentation:
            freeze_module(self.segmentation_model)

    def pool_fc(self, x):
        m = self.classification_model
        x = m.global_pool(x)
        if m.drop_rate:
            x = F.dropout(x, p=float(m.drop_rate), training=m.training)
        x = m.fc(x)
        return x

    def forward(self, x):
        if self.freeze_segmentation:
            self.segmentation_model.eval()
            with torch.no_grad():
                mask = self.segmentation_model(x)
        else:
            mask = self.segmentation_model(x)
        
        if self.concat:
            mask = torch.cat([x, mask], axis=1)

        if self.return_feature:
            feature = self.classification_model.forward_features(mask)
            output = self.pool_fc(feature)
        else:
            output = self.classification_model(mask)

        if self.return_mask and self.return_feature:
            return mask, feature, output
        elif self.return_mask:
            return mask, output
        elif self.return_feature:
            return feature, output
        else:
            return output
