from .base_mapper import BaseMapper, MAPPERS
from mmdet3d.models.builder import build_backbone, build_head, build_loss

@MAPPERS.register_module()
class HDMapNet_Semantic(BaseMapper):

    def __init__(self,
                 backbone_cfg,
                 head_cfg,
                 loss_cfg,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        super().__init__()

        self.backbone = build_backbone(backbone_cfg)
        self.head = build_head(head_cfg)
        self.loss = build_loss(loss_cfg)

    def forward_train(self, **kwargs):
        '''
        Args:
            img: torch.Tensor of shape [B, N, 3, H, W]
                N: number of cams
            vectors: list[list[Tuple(lines, length, label)]]
                - lines: np.array of shape [num_points, 2]. 
                - length: int
                - label: int
                len(vectors) = batch_size
                len(vectors[_b]) = num of lines in sample _b
            img_metas: 
                img_metas['lidar2img']: [B, N, 4, 4]

        Out:
            loss, log_vars, num_sample
        '''

        img = kwargs['img']
        semantic_mask_gt = kwargs['semantic_mask']
        img_metas = kwargs['img_metas']
        
        bev_feature = self.backbone(img, img_metas)

        out = self.head(bev_feature)

        loss = self.loss(out, semantic_mask_gt.float())
        log_vars = {'seg_loss': loss.item()}

        num_sample = img.size(0)
        return loss, log_vars, num_sample

    def forward_test(self, **kwargs):
        '''
            inference pipeline
        '''

        img = kwargs['img']
        bs = img.size(0)
        img_metas = kwargs['img_metas']

        bev_feature = self.backbone(img, img_metas)
        
        out = self.head(bev_feature)

        result_list = []
        for i in range(bs):
            result = {
                'semantic_mask': out[i].cpu(),
                'token': img_metas[i]['token']
            }
            result_list.append(result)

        return result_list
