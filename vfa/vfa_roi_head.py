from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from mmcv.utils import ConfigDict
from mmdet.core import bbox2roi
from mmdet.models.builder import HEADS
from mmfewshot.detection.models.roi_heads.meta_rcnn_roi_head import MetaRCNNRoIHead

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence

import torch
from torch.autograd import Function


class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                                    inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
                           '`VectorQuantization`. The function `VectorQuantization` '
                           'is not differentiable. Use `VectorQuantizationStraightThrough` '
                           'if you want a straight-through estimator of the gradient.')


class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vq(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0,
                                           index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous()
                                   .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)

vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply
__all__ = [vq, vq_st]


def to_scalar(arr):
    if type(arr) == list:
        return [x.item() for x in arr]
    else:
        return arr.item()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1. / K, 1. / K)
        self.latent2one = nn.Conv1d(512, 1, 1)

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 1).contiguous()  # 当调用contiguous()时，会强制拷贝一份tensor，让它的布局和从头创建的一模一样，但是两个tensor完全没有联系。
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 1).contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        # z_q_x = z_q_x_.permute(0, 2, 1).contiguous()
        z_q_x_onehot = self.latent2one(z_q_x_)

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
                                               dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        # z_q_x_bar = z_q_x_bar_.permute(0, 2, 1).contiguous()
        z_q_x_bar_onehot = self.latent2one(z_q_x_bar_)

        return z_q_x_, z_q_x_onehot, z_q_x_bar_, z_q_x_bar_onehot


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(dim, dim, 3, 1, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(True),
            nn.Conv1d(dim, dim, 1),
            nn.BatchNorm1d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


class VectorQuantizedVAE(nn.Module):
    def __init__(self, input_dim, dim, K=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(True),
            nn.Conv1d(dim, dim, 4, 2, 1),
            ResBlock(dim),
            ResBlock(dim),
        )

        self.codebook = VQEmbedding(K, dim)

        self.decoder = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            nn.ReLU(True),
            nn.ConvTranspose1d(dim, dim, 4, 2, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(True),
            nn.ConvTranspose1d(dim, input_dim, 4, 2, 1),
            nn.Tanh()
        )

        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def loss_function(self, images, x_tilde, z_e_x, z_q_x):
        # Reconstruction loss
        loss_recons = F.mse_loss(x_tilde, images)
        # Vector quantization objective
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        # Commitment objective
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

        loss = loss_recons + loss_vq + 0.25 * loss_commit
        return {'loss_vqvae': loss}

    def forward(self, x):
        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x_st_onehot, z_q_x, z_q_x_onehot = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x, z_q_x_onehot


# class VAE(nn.Module):
#
#     def __init__(self,
#                  in_channels: int,
#                  latent_dim: int,
#                  hidden_dim: int) -> None:
#         super(VAE, self).__init__()
#
#         self.latent_dim = latent_dim
#
#         self.encoder = nn.Sequential(
#             nn.Linear(in_channels, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.LeakyReLU()
#         )
#         self.fc_mu = nn.Linear(hidden_dim, latent_dim)
#         self.fc_var = nn.Linear(hidden_dim, latent_dim)
#
#         self.decoder_input = nn.Linear(latent_dim, hidden_dim)
#
#         self.decoder = nn.Sequential(
#             nn.Linear(hidden_dim, in_channels),
#             nn.BatchNorm1d(in_channels),
#             nn.Sigmoid()
#         )
#
#     def encode(self, input: Tensor) -> List[Tensor]:
#         result = self.encoder(input)
#
#         mu = self.fc_mu(result)
#         log_var = self.fc_var(result)
#
#         return [mu, log_var]
#
#     def decode(self, z: Tensor) -> Tensor:
#
#         z = self.decoder_input(z)
#         z_out = self.decoder(z)
#         return z_out
#
#     def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return eps * std + mu, std + mu
#
#     def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
#         mu, log_var = self.encode(input)
#         z, z_inv = self.reparameterize(mu, log_var)
#         z_out = self.decode(z)
#
#         return [z_out, z_inv, input, mu, log_var]
#
#     def loss_function(self, input, rec, mu, log_var, kld_weight=0.00025) -> dict:
#         recons_loss = F.mse_loss(rec, input)
#
#         kld_loss = torch.mean(-0.5 * torch.sum(1 +
#                               log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
#
#         loss = recons_loss + kld_weight * kld_loss
#
#         return {'loss_vae': loss}

# VQVAE

# model = VectorQuantizedVAE(num_channels, args.hidden_size, args.k).to(args.device)
#
# # num_channels  图片的channel数
# # args.hidden_size  size of the latent vectors==256
# # args.k  number of latent vectors==512
# # args.beta  0.25
#
# def train(data_loader, model, optimizer, args):
#     for images, _ in data_loader:
#         images = images.to(args.device)
#
#         optimizer.zero_grad()
#         x_tilde, z_e_x, z_q_x = model(images)
#
#         # Reconstruction loss
#         loss_recons = F.mse_loss(x_tilde, images)
#         # Vector quantization objective
#         loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
#         # Commitment objective
#         loss_commit = F.mse_loss(z_e_x, z_q_x.detach())
#
#         loss = loss_recons + loss_vq + args.beta * loss_commit
#         loss.backward()


@HEADS.register_module()
class VFARoIHead(MetaRCNNRoIHead):

    def __init__(self, vae_dim=2048, *args, **kargs) -> None:
        super().__init__(*args, **kargs)

        # self.vae = VAE(vae_dim, vae_dim, vae_dim)  # VQVAE
        self.vae = VectorQuantizedVAE(1, 2048, 512)

        # model = VectorQuantizedVAE(num_channels, args.hidden_size, args.k).to(args.device)
        # num_channels  图片的channel数
        # args.hidden_size  size of the latent vectors==256
        # args.k  number of latent vectors==512
        # args.beta  0.25

    def _bbox_forward_train(self, query_feats: List[Tensor],
                            support_feats: List[Tensor],
                            sampling_results: object,
                            query_img_metas: List[Dict],
                            query_gt_bboxes: List[Tensor],
                            query_gt_labels: List[Tensor],
                            support_gt_labels: List[Tensor]) -> Dict:
        """Forward function and calculate loss for box head in training.

        Args:
            query_feats (list[Tensor]): List of query features, each item
                with shape (N, C, H, W).
            support_feats (list[Tensor]): List of support features, each item
                with shape (N, C, H, W).
            sampling_results (obj:`SamplingResult`): Sampling results.
            query_img_metas (list[dict]): List of query image info dict where
                each dict has: 'img_shape', 'scale_factor', 'flip', and may
                also contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            query_gt_bboxes (list[Tensor]): Ground truth bboxes for each query
                image with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y]
                format.
            query_gt_labels (list[Tensor]): Class indices corresponding to
                each box of query images.
            support_gt_labels (list[Tensor]): Class indices corresponding to
                each box of support images.

        Returns:
            dict: Predicted results and losses.
        """
        query_rois = bbox2roi([res.bboxes for res in sampling_results])
        query_roi_feats = self.extract_query_roi_feat(query_feats, query_rois)
        support_feat = self.extract_support_feats(support_feats)[0]

        # support_feat_rec, support_feat_inv, _, mu, log_var = self.vae(
        #     support_feat)   # VQVAE的输入和输出
        support_feat_3 = torch.unsqueeze(support_feat, 1)
        support_feat_rec, z_e_x, support_feat_inv, support_feat_inv_onehot = self.vae(support_feat_3)
        support_feat_inv_onehot = torch.squeeze(support_feat_inv_onehot, 1)
        support_feat_rec_2 = torch.squeeze(support_feat_rec, 1)
        # x_tilde, z_e_x, z_q_x = model(images)

        bbox_targets = self.bbox_head.get_targets(sampling_results,
                                                  query_gt_bboxes,
                                                  query_gt_labels,
                                                  self.train_cfg)
        (labels, label_weights, bbox_targets, bbox_weights) = bbox_targets
        loss_bbox = {'loss_cls': [], 'loss_bbox': [], 'acc': []}
        batch_size = len(query_img_metas)
        num_sample_per_imge = query_roi_feats.size(0) // batch_size
        bbox_results = None
        for img_id in range(batch_size):
            start = img_id * num_sample_per_imge
            end = (img_id + 1) * num_sample_per_imge
            # class agnostic aggregation
            # random_index = np.random.choice(
            #     range(query_gt_labels[img_id].size(0)))
            # random_query_label = query_gt_labels[img_id][random_index]
            random_index = np.random.choice(
                range(len(support_gt_labels)))
            random_query_label = support_gt_labels[random_index]
            for i in range(support_feat.size(0)):
                if support_gt_labels[i] == random_query_label:
                    bbox_results = self._bbox_forward(
                        query_roi_feats[start:end],
                        support_feat_inv_onehot[i].sigmoid().unsqueeze(0))
                    single_loss_bbox = self.bbox_head.loss(
                        bbox_results['cls_score'], bbox_results['bbox_pred'],
                        query_rois[start:end], labels[start:end],
                        label_weights[start:end], bbox_targets[start:end],
                        bbox_weights[start:end])
                    for key in single_loss_bbox.keys():
                        loss_bbox[key].append(single_loss_bbox[key])
        if bbox_results is not None:
            for key in loss_bbox.keys():
                if key == 'acc':
                    loss_bbox[key] = torch.cat(loss_bbox['acc']).mean()
                else:
                    loss_bbox[key] = torch.stack(
                        loss_bbox[key]).sum() / batch_size

        # meta classification loss
        if self.bbox_head.with_meta_cls_loss:
            # input support feature classification
            meta_cls_score = self.bbox_head.forward_meta_cls(support_feat_rec_2)
            meta_cls_labels = torch.cat(support_gt_labels)
            loss_meta_cls = self.bbox_head.loss_meta(
                meta_cls_score, meta_cls_labels,
                torch.ones_like(meta_cls_labels))
            loss_bbox.update(loss_meta_cls)

        # loss_vae = self.vae.loss_function(
        #     support_feat, support_feat_rec, mu, log_var)   # VAE的loss函数
        loss_vae = self.vae.loss_function(self, support_feat_3, support_feat_rec, z_e_x, support_feat_inv)

        loss_bbox.update(loss_vae)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _bbox_forward(self, query_roi_feats: Tensor,
                      support_roi_feats: Tensor) -> Dict:
        """Box head forward function used in both training and testing.

        Args:
            query_roi_feats (Tensor): Query roi features with shape (N, C).
            support_roi_feats (Tensor): Support features with shape (1, C).

        Returns:
             dict: A dictionary of predicted results.
        """
        # feature aggregation
        roi_feats = self.aggregation_layer(
            query_feat=query_roi_feats.unsqueeze(-1).unsqueeze(-1),
            support_feat=support_roi_feats.view(1, -1, 1, 1))[0]
        cls_score, bbox_pred = self.bbox_head(
            roi_feats.squeeze(-1).squeeze(-1), query_roi_feats)
        bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred)
        return bbox_results

    def simple_test_bboxes(
            self,
            query_feats: List[Tensor],
            support_feats_dict: Dict,
            query_img_metas: List[Dict],
            proposals: List[Tensor],
            rcnn_test_cfg: ConfigDict,
            rescale: bool = False) -> Tuple[List[Tensor], List[Tensor]]:
        """Test only det bboxes without augmentation.

        Args:
            query_feats (list[Tensor]): Features of query image,
                each item with shape (N, C, H, W).
            support_feats_dict (dict[int, Tensor]) Dict of support features
                used for inference only, each key is the class id and value is
                the support template features with shape (1, C).
            query_img_metas (list[dict]): list of image info dict where each
                dict has: `img_shape`, `scale_factor`, `flip`, and may also
                contain `filename`, `ori_shape`, `pad_shape`, and
                `img_norm_cfg`. For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            proposals (list[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: Each tensor in first list
                with shape (num_boxes, 4) and with shape (num_boxes, )
                in second list. The length of both lists should be equal
                to batch_size.
        """
        img_shapes = tuple(meta['img_shape'] for meta in query_img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in query_img_metas)

        rois = bbox2roi(proposals)

        query_roi_feats = self.extract_query_roi_feat(query_feats, rois)
        cls_scores_dict, bbox_preds_dict = {}, {}
        num_classes = self.bbox_head.num_classes
        for class_id in support_feats_dict.keys():
            support_feat = support_feats_dict[class_id]

            # support_feat_rec, support_feat_inv, _, mu, log_var = self.vae(   # VQVAE的输出
            #     support_feat)
            support_feat_3 = torch.unsqueeze(support_feat, 1)
            support_feat_rec, z_e_x, support_feat_inv, support_feat_inv_onehot = self.vae(support_feat_3)
            support_feat_inv_onehot = torch.squeeze(support_feat_inv_onehot, 1)

            bbox_results = self._bbox_forward(
                query_roi_feats, support_feat_inv_onehot.sigmoid())  # support变量
            cls_scores_dict[class_id] = \
                bbox_results['cls_score'][:, class_id:class_id + 1]
            bbox_preds_dict[class_id] = \
                bbox_results['bbox_pred'][:, class_id * 4:(class_id + 1) * 4]
            # the official code use the first class background score as final
            # background score, while this code use average of all classes'
            # background scores instead.
            if cls_scores_dict.get(num_classes, None) is None:
                cls_scores_dict[num_classes] = \
                    bbox_results['cls_score'][:, -1:]
            else:
                cls_scores_dict[num_classes] += \
                    bbox_results['cls_score'][:, -1:]
        cls_scores_dict[num_classes] /= len(support_feats_dict.keys())
        cls_scores = [
            cls_scores_dict[i] if i in cls_scores_dict.keys() else
            torch.zeros_like(cls_scores_dict[list(cls_scores_dict.keys())[0]])
            for i in range(num_classes + 1)
        ]
        bbox_preds = [
            bbox_preds_dict[i] if i in bbox_preds_dict.keys() else
            torch.zeros_like(bbox_preds_dict[list(bbox_preds_dict.keys())[0]])
            for i in range(num_classes)
        ]
        cls_score = torch.cat(cls_scores, dim=1)
        bbox_pred = torch.cat(bbox_preds, dim=1)

        # split batch bbox prediction back to each image
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        bbox_pred = bbox_pred.split(num_proposals_per_img, 0)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels
