from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging
import torch.nn as nn

from datasets.bases import tokenize
from jisuan_PCA import jisuan_PCA_2
import time
from keshihua_gap import reduce_and_visualize
from model.build_model import obj_mlp
from utils.simple_tokenizer import SimpleTokenizer
import torch

def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices



class Evaluator():
    def __init__(self, img_loader, txt_loader):
        self.img_loader = img_loader  # gallery
        self.txt_loader = txt_loader  # query
        self.logger = logging.getLogger("LG-MGC.eval")

    def _compute_embedding(self, model, epoch):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats, text_embeddings, image_embeddings, image_n_top_ks, text_n_top_ks , image_sm,text_sm = [], [], [], [], [], [], [], [],[],[]
        # text
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat, text_embedding = model.encode_text_cross(caption)
                text_n_top_k = model.encode_n_topk(text_feat, text_embedding)
                text_topk_sm = model.text_poolm(text_embedding)

            qids.append(pid.view(-1))  # flatten
            qfeats.append(text_feat)
            text_embeddings.append(text_embedding)
            text_n_top_ks.append(text_n_top_k)
            text_sm.append(text_topk_sm)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)
        text_embeddings = torch.cat(text_embeddings, 0)
        text_n_top_ks = torch.cat(text_n_top_ks, 0)
        text_sm = torch.cat(text_sm, 0)

        # image
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat, image_embedding = model.encode_image_cross(img)
                image_embedding = image_embedding[:, 1:, :].float()
                image_n_top_k = model.encode_n_topk(img_feat, image_embedding)
                image_topk_sm = model.image_poolm(image_embedding)


            gids.append(pid.view(-1))  # flatten
            gfeats.append(img_feat)
            image_embeddings.append(image_embedding)
            image_n_top_ks.append(image_n_top_k)
            image_sm.append(image_topk_sm)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)
        image_embeddings = torch.cat(image_embeddings, 0)
        image_n_top_ks = torch.cat(image_n_top_ks, 0)
        image_sm = torch.cat(image_sm, 0)

        combined_i_tensor = torch.cat((gfeats, image_n_top_ks), dim=1)
        combined_t_tensor = torch.cat((qfeats, text_n_top_ks), dim=1)

        combined_i_tensor_sm = torch.cat((gfeats, image_sm), dim=1)
        combined_t_tensor_sm = torch.cat((qfeats, text_sm), dim=1)

        return combined_t_tensor, combined_i_tensor, qids, gids , combined_t_tensor_sm,combined_i_tensor_sm , qfeats ,gfeats


    def eval(self, model, epoch, i2t_metric=True):
        start1 = time.time()

        qfeats, gfeats, qids, gids , qfeats_sm, gfeats_sm , q_clss,g_clss = self._compute_embedding(model, epoch)

        qfeats = F.normalize(qfeats, p=2, dim=1)  # text features
        gfeats = F.normalize(gfeats, p=2, dim=1)  # image features

        qfeats_sm = F.normalize(qfeats_sm, p=2, dim=1)  # text features
        gfeats_sm = F.normalize(gfeats_sm, p=2, dim=1)  # image features

        q_clss = F.normalize(q_clss, p=2, dim=1)  # text features
        g_clss = F.normalize(g_clss, p=2, dim=1)  # image features

        start = time.time()

        c_qz = 1
        t_qz = 0
        s_qz = 0

        similarity_t = qfeats @ gfeats.t()
        similarity_sm = qfeats_sm @ gfeats_sm.t()
        similarity_cls = q_clss @ g_clss.t()
        similarity = c_qz * similarity_cls + t_qz * similarity_t + s_qz * similarity_sm

        end = time.time()
        print("calculate similarity time1: {}".format(end - start))
        print("calculate similarity time2: {}".format(end - start1))

        t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table.add_row(['t2i', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])

        if i2t_metric:
            i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=similarity.t(), q_pids=gids, g_pids=qids, max_rank=10,
                                                 get_mAP=True)
            i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
            table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])

        end1 = time.time()
        print("calculate similarity time3: {}".format(end1 - start))

        table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        self.logger.info('\n' + str(table))

        ir_mean = (t2i_cmc[0] + t2i_cmc[4] + t2i_cmc[9]) / 3
        tr_mean = (i2t_cmc[0] + i2t_cmc[4] + i2t_cmc[9]) / 3
        r_mean = (tr_mean + ir_mean) / 2


        return r_mean
