# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, interpolate
from detectron2.structures import Instances, heatmaps_to_keypoints
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

_TOTAL_SKIPPED = 0

ROI_KEYPOINT_HEAD_REGISTRY = Registry("ROI_KEYPOINT_HEAD")
ROI_KEYPOINT_HEAD_REGISTRY.__doc__ = """
Registry for keypoint heads, which make keypoint predictions from per-region features.
The registered object will be called with `obj(cfg, input_shape)`.
"""

def build_keypoint_head(cfg, input_shape):
    """
    Build a keypoint head from `cfg.MODEL.ROI_KEYPOINT_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_KEYPOINT_HEAD.NAME
    return ROI_KEYPOINT_HEAD_REGISTRY.get(name)(cfg, input_shape)

@torch.no_grad()
def keypoints_to_heatmap_mod(
    keypoints, rois, heatmap_size
) :
    """
    Encode keypoint locations into a target heatmap for use in SoftmaxWithLoss across space.
    Maps keypoints from the half-open interval [x1, x2) on continuous image coordinates to the
    closed interval [0, heatmap_size - 1] on discrete image coordinates. We use the
    continuous-discrete conversion from Heckbert 1990 ("What is the coordinate of a pixel?"):
    d = floor(c) and c = d + 0.5, where d is a discrete coordinate and c is a continuous coordinate.
    Arguments:
        keypoints: tensor of keypoint locations in of shape (N, K, 3).
        rois: Nx4 tensor of rois in xyxy format
        heatmap_size: integer side length of square heatmap.
    Returns:
        heatmaps: A tensor of shape (N, K, 2) containing x,y coordinates for each keypoint with coordinates
            in the range [0, heatmap_size - 1] for each keypoint in the input.
        valid: A tensor of shape (N, K) containing whether each keypoint is in
            the roi or not.
    """
    N,K,_ = keypoints.shape
    heatmaps = keypoints.new_zeros(N,K,2)
    
    if rois.numel() == 0:
        return rois.new().long(), rois.new().long()
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = heatmap_size / (rois[:, 2] - rois[:, 0])
    scale_y = heatmap_size / (rois[:, 3] - rois[:, 1])

    offset_x = offset_x[:, None]
    offset_y = offset_y[:, None]
    scale_x = scale_x[:, None]
    scale_y = scale_y[:, None]

    x = keypoints[..., 0]
    y = keypoints[..., 1]

    x_boundary_inds = x == rois[:, 2][:, None]
    y_boundary_inds = y == rois[:, 3][:, None]

    x = (x - offset_x) * scale_x
    x = x.floor().long()
    y = (y - offset_y) * scale_y
    y = y.floor().long()

    x[x_boundary_inds] = heatmap_size - 1
    y[y_boundary_inds] = heatmap_size - 1

    valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
    vis = keypoints[..., 2] > 0
    valid = (valid_loc & vis).long()

    assert int(torch.max(x*valid)) < heatmap_size, "current max is {}".format(int(torch.max(x*valid)))
    assert int(torch.max(y*valid)) < heatmap_size, "current max is {}".format(int(torch.max(y*valid)))

    heatmaps[:,:,0] = x
    heatmaps[:,:,1] = y 
    """
    Check if below works, will probably require valid.unsqueeze(-1)
    """
    heatmaps = heatmaps * (valid.unsqueeze(-1))

    return heatmaps, valid


def keypoint_rcnn_loss(pred_keypoint_logits, instances, normalizer):
    """
    @Nihal
    Arguments:
        pred_keypoint_logits (Tensor): A tensor of shape (N, K, S, S) where N is the total number
            of instances in the batch, K is the number of keypoints, and S is the side length
            of the keypoint heatmap. The values are spatial logits.
        instances (list[Instances]): A list of M Instances, where M is the batch size.
            These instances are predictions from the model
            that are in 1:1 correspondence with pred_keypoint_logits.
            Each Instances should contain a `gt_keypoints` field containing a `structures.Keypoint`
            instance.
        normalizer (float): Normalize the loss by this amount.
            If not specified, we normalize by the number of visible keypoints in the minibatch.
    Returns a scalar tensor containing the loss.
    """
    heatmaps = []
    valid = []

    keypoint_side_len = pred_keypoint_logits.shape[2]
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        keypoints = instances_per_image.gt_keypoints
        heatmaps_per_image, valid_per_image = keypoints_to_heatmap_mod(keypoints.tensor,
            instances_per_image.proposal_boxes.tensor, keypoint_side_len
        )
        heatmaps.append(heatmaps_per_image)
        #print("number of valid keypoints is {}".format(torch.sum(valid_per_image)))
        valid.append(valid_per_image.view(-1))

    if len(heatmaps):
        keypoint_targets = cat(heatmaps, dim=0) #shape should be N,K,2 
        valid = cat(valid, dim=0).to(dtype=torch.uint8)
        valid = torch.nonzero(valid).squeeze(1)

    keypoint_targets = keypoint_targets.view(-1,2)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if len(heatmaps) == 0 or valid.numel() == 0:
        global _TOTAL_SKIPPED
        _TOTAL_SKIPPED += 1
        storage = get_event_storage()
        storage.put_scalar("kpts_num_skipped_batches", _TOTAL_SKIPPED, smoothing_hint=False)
        return pred_keypoint_logits.sum() * 0
    """
    @Nihal
    softargmax_kp is a tensor of shape(N), where N is no. of instances. Each entry in softargmax_kp is the predicted keypoint position in [0,H*W).
    This predicted keypoint position is found using the softargmax function.
    keypoint_targets contains the ground truth keypoint position, belonging to [0,H*W)
    Thus we can now use l1 or l2 loss between the predicted and the target keypoints.
    """

    N, K, H, W = pred_keypoint_logits.shape
    #print("number of instances considered for KP loss {}".format(N))
    pred_keypoint_logits = pred_keypoint_logits.view(N * K, H * W)
    pred_keypoint_softmax = F.softmax(pred_keypoint_logits,dim=1)
    kp_idx_h = torch.arange(H,device='cuda').unsqueeze(-1).float()
    kp_idx_w = torch.arange(W,device='cuda').float()

    pred_keypoint_softmax = pred_keypoint_softmax.view(N,K,H,W)
    pred_kp_x = ((pred_keypoint_softmax*kp_idx_w).view(N*K,H*W)).sum(dim=1)
    pred_kp_y = ((pred_keypoint_softmax*kp_idx_h).view(N*K,H*W)).sum(dim=1)

    pred_kp_x = pred_kp_x.unsqueeze(-1)
    pred_kp_y = pred_kp_y.unsqueeze(-1)
    
    pred_kps = torch.cat((pred_kp_x,pred_kp_y),-1)#shape N*K,2

    print("predicted keypoints are")
    print(pred_kps[valid])

    print("targets are")
    print(keypoint_targets[valid])

    if not list(pred_kps.shape) == list(keypoint_targets.shape):
        print("pred_kps shape {}".format(list(pred_kps.shape)))
        print("keypoint targets shape {}".format(list(keypoint_targets.shape)))
        raise RuntimeError('targets and predictions should be of same shape')

    keypoint_targets = keypoint_targets.float()
    keypoint_loss = F.l1_loss(
        pred_kps[valid], keypoint_targets[valid], reduction="mean"
    )

    num_valid_keypoints = valid.numel()
    print("num_valid_keypoints {}".format(num_valid_keypoints))
    storage = get_event_storage()
    storage.put_scalar("num_valid_kp_per_instance", num_valid_keypoints/N)

    return keypoint_loss

def heatmaps_to_rescaled_keypoints(maps: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
    """
    Extract predicted keypoint locations from heatmaps.
    Args:
        maps (Tensor): (#ROIs, #keypoints, POOL_H, POOL_W). The predicted heatmap of logits for
            each ROI and each keypoint.
        rois (Tensor): (#ROIs, 4). The box of each ROI.
    Returns:
        Tensor of shape (#ROIs, #keypoints, 4) with the last dimension corresponding to
        (x, y, logit, score) for each keypoint.
    When converting discrete pixel indices in an NxN image to a continuous keypoint coordinate,
    we maintain consistency with :meth:`Keypoints.to_heatmap` by using the conversion from
    Heckbert 1990: c = d + 0.5, where d is a discrete coordinate and c is a continuous coordinate.
    """
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]

    offset_x = offset_x.unsqueeze(-1) #shape N,1
    offset_y = offset_y.unsqueeze(-1) #shape N,1

    widths = (rois[:, 2] - rois[:, 0]).clamp(min=1) 
    heights = (rois[:, 3] - rois[:, 1]).clamp(min=1)

    widths = widths.unsqueeze(-1) #shape N,1
    heights = heights.unsqueeze(-1) #shape N,1

    N,K,H,W = maps.shape
    
    preds = maps.new_zeros(N,K,3)

    

    pred_keypoint_logits = maps.view(N * K, H * W)
    
    pred_keypoint_logits = pred_keypoint_logits.view(N * K, H * W)
    pred_keypoint_softmax = F.softmax(pred_keypoint_logits,dim=1)
    kp_idx_h = torch.arange(H,device='cuda').unsqueeze(-1).float()
    kp_idx_w = torch.arange(W,device='cuda').float()

    pred_keypoint_softmax = pred_keypoint_softmax.view(N,K,H,W)
    pred_kp_x = ((pred_keypoint_softmax*kp_idx_w).view(N*K,H*W)).sum(dim=1)
    pred_kp_y = ((pred_keypoint_softmax*kp_idx_h).view(N*K,H*W)).sum(dim=1)


    pred_kp_x = pred_kp_x.view(N,K)
    pred_kp_y = pred_kp_y.view(N,K)

    pred_kp_x = (pred_kp_x * widths/W) + offset_x
    pred_kp_y = (pred_kp_y * heights/H) + offset_y

    preds[:,:,0] = pred_kp_x
    preds[:,:,1] = pred_kp_y
    preds[:,:,2] = torch.ones(N,K)

    print("x offsets are {}".format(offset_x))
    print("y offsets are {}".format(offset_y))
    print("widths are {}".format(widths))
    print("heights are {}".format(heights))

    print(preds[0:2])
    
    return preds

def keypoint_rcnn_inference(pred_keypoint_logits, pred_instances):
    """
    Post process each predicted keypoint heatmap in `pred_keypoint_logits` into (x, y, score)
        and add it to the `pred_instances` as a `pred_keypoints` field.
    Args:
        pred_keypoint_logits (Tensor): A tensor of shape (R, K, S, S) where R is the total number
           of instances in the batch, K is the number of keypoints, and S is the side length of
           the keypoint heatmap. The values are spatial logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images.
    Returns:
        None. Each element in pred_instances will contain an extra "pred_keypoints" field.
            The field is a tensor of shape (#instance, K, 3) where the last
            dimension corresponds to (x, y, score).
            The scores are larger than 0.
    """
    # flatten all bboxes from all images together (list[Boxes] -> Rx4 tensor)
    bboxes_flat = cat([b.pred_boxes.tensor for b in pred_instances], dim=0)

    keypoint_results = heatmaps_to_rescaled_keypoints(pred_keypoint_logits.detach(), bboxes_flat.detach())
    num_instances_per_image = [len(i) for i in pred_instances]
    keypoint_results = keypoint_results[:, :, [0, 1, 2]].split(num_instances_per_image, dim=0)

    for keypoint_results_per_image, instances_per_image in zip(keypoint_results, pred_instances):
        # keypoint_results_per_image is (num instances)x(num keypoints)x(x, y, score)
        instances_per_image.pred_keypoints = keypoint_results_per_image


class BaseKeypointRCNNHead(nn.Module):
    """
    Implement the basic Keypoint R-CNN losses and inference logic.
    """

    def __init__(self, cfg, input_shape):
        super().__init__()
        # fmt: off
        self.loss_weight                    = cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT
        self.normalize_by_visible_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS  # noqa
        self.num_keypoints                  = cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS
        batch_size_per_image                = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        positive_sample_fraction            = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        # fmt: on
        self.normalizer_per_img = (
            self.num_keypoints * batch_size_per_image * positive_sample_fraction
        )

    def forward(self, x, instances: List[Instances]):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.
        Returns:
            A dict of losses if in training. The predicted "instances" if in inference.
        """
        x = self.layers(x)
        if self.training:
            num_images = len(instances)
            normalizer = (
                None
                if self.normalize_by_visible_keypoints
                else num_images * self.normalizer_per_img
            )
            return {
                "loss_keypoint": keypoint_rcnn_loss(x, instances, normalizer=normalizer)
                * self.loss_weight
            }
        else:
            keypoint_rcnn_inference(x, instances)
            return instances

    def layers(self, x):
        """
        Neural network layers that makes predictions from regional input features.
        """
        raise NotImplementedError


@ROI_KEYPOINT_HEAD_REGISTRY.register()
class KRCNNConvDeconvUpsampleHead(BaseKeypointRCNNHead):
    """
    A standard keypoint head containing a series of 3x3 convs, followed by
    a transpose convolution and bilinear interpolation for upsampling.
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            conv_dims: an iterable of output channel counts for each conv in the head
                         e.g. (512, 512, 512) for three convs outputting 512 channels.
            num_keypoints: number of keypoint heatmaps to predicts, determines the number of
                           channels in the final output.
        """
        super().__init__(cfg, input_shape)

        # fmt: off
        # default up_scale to 2 (this can eventually be moved to config)
        up_scale      = 2
        conv_dims     = cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_DIMS
        num_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS
        in_channels   = input_shape.channels
        # fmt: on

        self.blocks = []
        for idx, layer_channels in enumerate(conv_dims, 1):
            module = Conv2d(in_channels, layer_channels, 3, stride=1, padding=1)
            self.add_module("conv_fcn{}".format(idx), module)
            self.blocks.append(module)
            in_channels = layer_channels

        deconv_kernel = 4
        self.score_lowres = ConvTranspose2d(
            in_channels, num_keypoints, deconv_kernel, stride=2, padding=deconv_kernel // 2 - 1
        )
        self.up_scale = up_scale

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def layers(self, x):
        for layer in self.blocks:
            x = F.relu(layer(x))
        x = self.score_lowres(x)
        x = interpolate(x, scale_factor=self.up_scale, mode="bilinear", align_corners=False)
        return x