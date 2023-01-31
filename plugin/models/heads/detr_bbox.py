import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear
from mmcv.runner import force_fp32
from torch.distributions.categorical import Categorical

from mmdet.core import multi_apply, reduce_mean
from mmdet.models import HEADS
from .detr_head import DETRMapFixedNumHead


@HEADS.register_module(force=True)
class DETRBboxHead(DETRMapFixedNumHead):

    def __init__(self, *args, canvas_size=(400, 200), discrete_output=True, separate_detect=True, 
    mode='xyxy', bbox_size=None, coord_dim=2, kp_coord_dim=2,
    **kwargs):
        self.canvas_size = canvas_size  # hard code

        self.separate_detect = separate_detect
        self.discrete_output = discrete_output
        self.bbox_size = 3 if mode=='sce' else 2
        if bbox_size is not None:
            self.bbox_size = bbox_size
        self.coord_dim = coord_dim  # for xyz
        self.kp_coord_dim = kp_coord_dim

        super(DETRBboxHead, self).__init__(*args, **kwargs)
        del self.canvas_size
        self.register_buffer('canvas_size', torch.tensor(canvas_size))
        self._init_embedding()
        
    def _init_embedding(self):

        # for bbox parameter xstart, ystart, xend, yend
        self.bbox_embedding = nn.Embedding(4, self.embed_dims)

        self.label_embed = nn.Embedding(
            self.num_classes, self.embed_dims)

        self.img_coord_embed = nn.Linear(2, self.embed_dims)

    def _init_branch(self,):
        """Initialize classification branch and regression branch of head."""
        
        # add sigmoid or not
        if self.separate_detect:
            if self.cls_out_channels == self.num_classes+1:
                self.cls_out_channels = 2
            else:
                self.cls_out_channels = 1

        fc_cls = Linear(self.embed_dims, self.cls_out_channels)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.LayerNorm(self.embed_dims))
            reg_branch.append(nn.ReLU())

        if self.discrete_output:
            reg_branch.append(nn.Linear(
                self.embed_dims, max(self.canvas_size), bias=True,))
        else:
            reg_branch.append(nn.Linear(
                self.embed_dims, self.bbox_size*self.coord_dim, bias=True,))

        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        num_pred = self.transformer.decoder.num_layers

        if self.iterative:
            fc_cls = _get_clones(fc_cls, num_pred)
            reg_branch = _get_clones(reg_branch, num_pred)

        self.pre_branches = nn.ModuleDict([
            ('cls', fc_cls),
            ('reg', reg_branch), ])

    def _prepare_context(self, batch, context):
        """Prepare class label and vertex context."""

        global_context_embedding = None
        if self.separate_detect:
            global_context_embedding = self.label_embed(batch['class_label'])

        # Image context
        if self.separate_detect:
            image_embeddings = assign_bev(
                context['bev_embeddings'], batch['batch_idx'])
        else:
            image_embeddings = context['bev_embeddings']

        image_embeddings = self.input_proj(
            image_embeddings)  # only change feature size

        # Pass images through encoder
        device = image_embeddings.device

        # Add 2D coordinate grid embedding
        B, C, H, W = image_embeddings.shape
        Ws = torch.linspace(-1., 1., W)
        Hs = torch.linspace(-1., 1., H)
        image_coords = torch.stack(
            torch.meshgrid(Hs, Ws), dim=-1).to(device)
        image_coord_embeddings = self.img_coord_embed(image_coords)

        image_embeddings += image_coord_embeddings[None].permute(0, 3, 1, 2)

        # Reshape spatial grid to sequence
        sequential_context_embeddings = image_embeddings.reshape(
            B, C, H, W)

        return (global_context_embedding, sequential_context_embeddings)

    def forward(self, batch, context, img_metas=None):
        '''
        Args:
            bev_feature (List[Tensor]): shape [B, C, H, W]
                feature in bev view
            img_metas
        Outs:
            preds_dict (Dict):
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_lines_preds (Tensor):
                    [nb_dec, bs, num_query, num_points, 2].
        '''

        (global_context_embedding, sequential_context_embeddings) =\
            self._prepare_context(batch, context)

        if self.separate_detect:
            query_embedding = self.query_embedding.weight[None] + \
                global_context_embedding[:, None]
        else:
            B = sequential_context_embeddings.shape[0]
            query_embedding = self.query_embedding.weight[None].repeat(B, 1, 1)

        x = sequential_context_embeddings
        B, C, H, W = x.shape

        masks = x.new_zeros((B, H, W))
        pos_embed = self.positional_encoding(masks)
        # outs_dec: [nb_dec, bs, num_query, embed_dim]
        outs_dec, _ = self.transformer(x, masks.type(torch.bool), query_embedding,
                                       pos_embed)

        outputs = []
        for i, query_feat in enumerate(outs_dec):
            outputs.append(self.get_prediction(query_feat))

        return outputs

    def get_prediction(self, query_feat):

        ocls = self.pre_branches['cls'](query_feat)

        if self.discrete_output:
            pos = []
            for i in range(4):
                pos_embeds = self.bbox_embedding.weight[i]
                _pos = self.pre_branches['reg'](query_feat+pos_embeds)
                pos.append(_pos)

            # # y mask
            # _vert_mask = torch.arange(logits.shape[-1], device=logits.device)
            # vertices_mask_y = (_vert_mask < self.canvas_size[1]+1)
            # logits[:,1::2] = logits[:,1::2]*vertices_mask_y - ~vertices_mask_y*1e9
            logits = torch.stack(pos, dim=-2)/1.
            lines = Categorical(logits=logits)
        else:
            lines = self.pre_branches['reg'](query_feat).sigmoid()
            lines = lines.unflatten(-1, (self.bbox_size, self.coord_dim))*self.canvas_size
            lines = lines.flatten(-2)

        return dict(
            lines=lines,  # [bs, num_query, 4, num_canvas_size]
            scores=ocls,  # [bs, num_query, num_class]
        )

    @force_fp32(apply_to=('score_pred', 'lines_pred', 'gt_lines'))
    def _get_target_single(self,
                           score_pred,
                           lines_pred,
                           gt_labels,
                           gt_lines,
                           gt_bboxes_ignore=None):
        """
            Compute regression and classification targets for one image.
            Outputs from a single decoder layer of a single feature level are used.
            Args:
                cls_score (Tensor): Box score logits from a single decoder layer
                    for one image. Shape [num_query, cls_out_channels].
                lines_pred (Tensor):
                    shape [num_query, num_points, 2].
                gt_lines (Tensor):
                    shape [num_gt, num_points, 2].
                gt_labels (torch.LongTensor)
                    shape [num_gt, ]
            Returns:
                tuple[Tensor]: a tuple containing the following for one image.
                    - labels (LongTensor): Labels of each image.
                        shape [num_query, 1]
                    - label_weights (Tensor]): Label weights of each image.
                        shape [num_query, 1]
                    - lines_target (Tensor): Lines targets of each image.
                        shape [num_query, num_points, 2]
                    - lines_weights (Tensor): Lines weights of each image.
                        shape [num_query, num_points, 2]
                    - pos_inds (Tensor): Sampled positive indices for each image.
                    - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_pred_lines = len(lines_pred)
        # assigner and sampler
        assign_result = self.assigner.assign(preds=dict(lines=lines_pred, scores=score_pred,),
                                             gts=dict(lines=gt_lines,
                                                      labels=gt_labels, ),
                                             gt_bboxes_ignore=gt_bboxes_ignore)
        sampling_result = self.sampler.sample(
            assign_result, lines_pred, gt_lines)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        pos_gt_inds = sampling_result.pos_assigned_gt_inds

        # label targets 0: foreground, 1: background
        if self.separate_detect:
            labels = gt_lines.new_full((num_pred_lines, ), 1, dtype=torch.long)
        else:
            labels = gt_lines.new_full(
                (num_pred_lines, ), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_lines.new_ones(num_pred_lines)

        # bbox targets since lines_pred's last dimension is the vocabulary
        # and ground truth dose not have this dimension.
        if self.discrete_output:
            lines_target = torch.zeros_like(lines_pred[..., 0]).long()
            lines_weights = torch.zeros_like(lines_pred[..., 0])
        else:
            lines_target = torch.zeros_like(lines_pred)
            lines_weights = torch.zeros_like(lines_pred)

        lines_target[pos_inds] = sampling_result.pos_gt_bboxes.type(
            lines_target.dtype)
        lines_weights[pos_inds] = 1.0

        n = lines_weights.sum(-1, keepdim=True)
        lines_weights = lines_weights / n.masked_fill(n == 0, 1)

        return (labels, label_weights, lines_target, lines_weights,
                pos_inds, neg_inds, pos_gt_inds)

    # @force_fp32(apply_to=('preds', 'gts'))
    def get_targets(self, preds, gts, gt_bboxes_ignore_list=None):
        """
            Compute regression and classification targets for a batch image.
            Outputs from a single decoder layer of a single feature level are used.
            Args:
                cls_scores_list (list[Tensor]): Box score logits from a single
                    decoder layer for each image with shape [num_query,
                    cls_out_channels].
                lines_preds_list (list[Tensor]): [num_query, num_points, 2].
                gt_lines_list (list[Tensor]): Ground truth lines for each image
                    with shape (num_gts, num_points, 2)
                gt_labels_list (list[Tensor]): Ground truth class indices for each
                    image with shape (num_gts, ).
                gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                    boxes which can be ignored for each image. Default None.
            Returns:
                tuple: a tuple containing the following targets.
                    - labels_list (list[Tensor]): Labels for all images.
                    - label_weights_list (list[Tensor]): Label weights for all \
                        images.
                    - lines_targets_list (list[Tensor]): Lines targets for all \
                        images.
                    - lines_weight_list (list[Tensor]): Lines weights for all \
                        images.
                    - num_total_pos (int): Number of positive samples in all \
                        images.
                    - num_total_neg (int): Number of negative samples in all \
                        images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'

        # format the inputs
        if self.separate_detect:
            bbox = [b[m] for b, m in zip(gts['bbox'], gts['bbox_mask'])]
            class_label = torch.zeros_like(gts['bbox_mask']).long()
            class_label = [b[m] for b, m in zip(class_label, gts['bbox_mask'])]
        else:
            class_label = gts['class_label']
            bbox = gts['bbox']

        if self.discrete_output:
            lines_pred = preds['lines'].logits
        else:
            lines_pred = preds['lines']
            bbox = [b.float() for b in bbox]

        (labels_list, label_weights_list,
         lines_targets_list, lines_weights_list,
         pos_inds_list, neg_inds_list,pos_gt_inds_list) = multi_apply(
             self._get_target_single,
             preds['scores'], lines_pred,
             class_label, bbox,
             gt_bboxes_ignore=gt_bboxes_ignore_list)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        new_gts = dict(
            labels=labels_list,
            label_weights=label_weights_list,
            bboxs=lines_targets_list,
            bboxs_weights=lines_weights_list,
        )

        return new_gts, num_total_pos, num_total_neg, pos_inds_list, pos_gt_inds_list

    # @force_fp32(apply_to=('preds', 'gts'))
    def loss_single(self,
                    preds: dict,
                    gts: dict,
                    gt_bboxes_ignore_list=None,
                    reduction='none'):
        """
            Loss function for outputs from a single decoder layer of a single
            feature level.
            Args:
                cls_scores (Tensor): Box score logits from a single decoder layer
                    for all images. Shape [bs, num_query, cls_out_channels].
                lines_preds (Tensor):
                    shape [bs, num_query, num_points, 2].
                gt_lines_list (list[Tensor]):
                    with shape (num_gts, num_points, 2)
                gt_labels_list (list[Tensor]): Ground truth class indices for each
                    image with shape (num_gts, ).
                gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                    boxes which can be ignored for each image. Default None.
            Returns:
                dict[str, Tensor]: A dictionary of loss components for outputs from
                    a single decoder layer.
        """

        # Get target for each sample
        new_gts, num_total_pos, num_total_neg, pos_inds_list, pos_gt_inds_list =\
            self.get_targets(preds, gts, gt_bboxes_ignore_list)

        # Batched all data
        for k, v in new_gts.items():
            new_gts[k] = torch.stack(v, dim=0)

        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                preds['scores'].new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        # Classification loss
        if self.separate_detect:
            loss_cls = self.bce_loss(
                preds['scores'], new_gts['labels'], new_gts['label_weights'], cls_avg_factor)
        else:
            # since the inputs needs the second dim is the class dim, we permute the prediction.
            cls_scores = preds['scores'].reshape(-1, self.cls_out_channels)
            cls_labels = new_gts['labels'].reshape(-1)
            cls_weights = new_gts['label_weights'].reshape(-1)
            loss_cls = self.loss_cls(
                cls_scores, cls_labels, cls_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # position NLL loss
        if self.discrete_output:
            loss_reg = -(preds['lines'].log_prob(new_gts['bboxs']) *
                         new_gts['bboxs_weights']).sum()/(num_total_pos)
        else:
            loss_reg = self.reg_loss(
                preds['lines'], new_gts['bboxs'], new_gts['bboxs_weights'], avg_factor=num_total_pos)

        loss_dict = dict(
            cls=loss_cls,
            reg=loss_reg,
        )

        return loss_dict, pos_inds_list, pos_gt_inds_list

    def bce_loss(self, logits, label, weights, cls_avg_factor):
        ''' binary ce plog(p) + (1-p)log(1-p)
            logits: B,n,1
            label:
        '''
        p = logits.squeeze(-1).sigmoid()

        pos_msk = label == 0
        neg_msk = ~pos_msk

        loss_cls = -(p.log()*pos_msk + (1-p).log()*neg_msk)

        loss_cls = (loss_cls * weights).sum()/cls_avg_factor

        return loss_cls

    def post_process(self, preds_dicts: list, **kwargs):
        '''
        Args:
            preds_dicts:
                scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                lines (Tensor):
                    [nb_dec, bs, num_query, bbox parameters(4)].
        Outs:
            ret_list (List[Dict]) with length as bs
                list of result dict for each sample in the batch
                XXX
        '''
        preds = preds_dicts[-1]

        batched_cls_scores = preds['scores']
        batched_lines_preds = preds['lines']
        batch_size = batched_cls_scores.size(0)
        device = batched_cls_scores.device

        result_dict = {
            'bbox': [],
            'scores': [],
            'labels': [],
            'bbox_flat': [],
            'lines_cls': [],
            'lines_bs_idx': [],
        }
        for i in range(batch_size):

            cls_scores = batched_cls_scores[i]
            det_preds = batched_lines_preds[i]
            max_num = self.max_lines

            if self.loss_cls.use_sigmoid:
                cls_scores = cls_scores.sigmoid()
                scores, valid_idx = cls_scores.view(-1).topk(max_num)
                det_labels = valid_idx % self.num_classes
                valid_idx = valid_idx // self.num_classes
                det_preds = det_preds[valid_idx]
            else:
                scores, det_labels = F.softmax(cls_scores, dim=-1)[..., :-1].max(-1)
                scores, valid_idx = scores.topk(max_num)
                det_preds = det_preds[valid_idx]
                det_labels = det_labels[valid_idx]

            nline = len(valid_idx)
            result_dict['bbox'].append(det_preds)
            result_dict['scores'].append(scores)
            result_dict['labels'].append(det_labels)
            result_dict['lines_bs_idx'].extend([i]*nline)

        # for down stream polyline
        _bboxs = torch.cat(result_dict['bbox'], dim=0)
        # quantize the data
        result_dict['bbox_flat'] = torch.round(_bboxs).type(torch.int32)

        result_dict['lines_cls'] = torch.cat(
            result_dict['labels'], dim=0).long()
        result_dict['lines_bs_idx'] = torch.tensor(
            result_dict['lines_bs_idx'], device=device).long()

        return result_dict


def assign_bev(feat, idx):
    return feat[idx]