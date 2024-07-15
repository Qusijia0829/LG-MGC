import argparse

from torch.nn import BatchNorm1d

from build_dalle import build_dalle

from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from .pooling import TopKPooling_1

class LG_MGC (nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size,
                                                                      args.stride_size)


        self.prior = build_dalle()

        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature)

        self.text_poolm = TopKPooling_1(self.args.pool_k, dim=1)
        self.image_poolm = TopKPooling_1(self.args.pool_k, dim=1)

        if 'pred' in self.current_task:
            self.vision_proj = self.base_model.visual.proj

        for param in self.prior.parameters():
            param.requires_grad = False

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        return x[:, 0, :].float()

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def encode_image_cross(self, image):
        x = self.base_model.encode_image(image)
        return x[:, 0, :].float(), x.float()

    def encode_text_cross(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float(), x.float()

    def encode_n_topk(self,i_feats,image_feats,image = None):
        bs_I = image_feats.shape[0]
        if image is not None:
            k = self.args.image_k
        else:
            k = self.args.text_k

        n_top_ks = []


        for batch_index in range(bs_I):
            i_feat_example = i_feats[batch_index]
            i_emb_example = image_feats[batch_index]
            i_similarities = F.cosine_similarity(i_feat_example.unsqueeze(0), i_emb_example, dim=1)
            i_top_K_indices = torch.topk(i_similarities, k=k, largest=False).indices
            i_n_top_K_tokens = i_emb_example[i_top_K_indices]
            n_top_k = torch.mean(i_n_top_K_tokens , dim=0)
            n_top_ks.append(n_top_k.float())

        n_top_ks = torch.stack(n_top_ks, dim=0)

        return n_top_ks

    def forward(self, batch ,epoch):
        ret = dict()

        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, text_feats = self.base_model(images, caption_ids)

        i_feats = image_feats[:, 0, :].float()
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
        text_feats = text_feats.float()
        image_feats = image_feats[:, 1:, :].float()
        image_n_top_k = self.encode_n_topk(i_feats, image_feats)
        text_n_top_k = self.encode_n_topk(t_feats, text_feats)
        combined_i_tensor = torch.cat((i_feats, image_n_top_k), dim=1)
        combined_t_tensor = torch.cat((t_feats, text_n_top_k), dim=1)
        pred_cls = self.prior.sample(caption_ids,timesteps=1)
        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        if 'itc' in self.current_task:
            ret.update({'itc_loss': objectives.compute_itc(i_feats, t_feats, logit_scale)})

        if 'pred' in self.current_task:

            pred_cls = pred_cls @ self.vision_proj.float()

            ret.update({'loss_pred_ip': objectives.compute_itc(i_feats, pred_cls, logit_scale)})
            ret.update({'loss_pred_tp': objectives.compute_itc(t_feats, pred_cls, logit_scale)})

        if 'top' in self.current_task:

            ret.update({'itc_combined_loss': objectives.compute_itc(combined_i_tensor, combined_t_tensor, logit_scale)})

        if 'topsm' in self.current_task:
            image_pool_topk = self.image_poolm(image_feats)
            text_pool_topk = self.text_poolm(text_feats)
            combined_i_tensor_pool = torch.cat((i_feats, image_pool_topk), dim=1)
            combined_t_tensor_pool = torch.cat((i_feats, text_pool_topk), dim=1)

            ret.update({'itc_topsm_loss': objectives.compute_itc(combined_i_tensor_pool, combined_t_tensor_pool, logit_scale)})


        return ret




def build_model(args, num_classes=11003):
    model = LG_MGC(args, num_classes)
    # covert model to fp16
    convert_weights(model.base_model)
    return model
