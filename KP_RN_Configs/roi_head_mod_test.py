import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch
import math
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from detectron2.modeling.backbone.resnet import BottleneckBlock, make_stage
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.roi_heads.keypoint_head import build_keypoint_head
from detectron2.modeling.roi_heads.roi_heads import select_foreground_proposals,select_proposals_with_visible_keypoints,ROI_HEADS_REGISTRY, ROIHeads


logger = logging.getLogger(__name__)

@ROI_HEADS_REGISTRY.register()
class RetinaNetKeypointROIHeads(ROIHeads):
	"""
	It's "standard" in a sense that there is no ROI transform sharing
	or feature sharing between tasks.
	The cropped rois go to separate branches (boxes and masks) directly.
	This way, it is easier to make separate abstractions for different branches.
	This class is used by most models, such as FPN and C5.
	To implement more models, you can subclass it and implement a different
	:meth:`forward()` or a head.
	"""

	def __init__(self, cfg, input_shape):
		super(RetinaNetKeypointROIHeads, self).__init__(cfg, input_shape)
		self._init_keypoint_head(cfg, input_shape)

	def _init_keypoint_head(self, cfg, input_shape):
		# fmt: off
		self.keypoint_on  = cfg.MODEL.KEYPOINT_ON
		if not self.keypoint_on:
			return
		pooler_resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
		pooler_scales     = tuple(1.0 / input_shape[k].stride for k in self.in_features)  # noqa
		sampling_ratio    = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
		pooler_type       = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
		# fmt: on

		in_channels = [input_shape[f].channels for f in self.in_features][0]

		self.keypoint_pooler = ROIPooler(
			output_size=pooler_resolution,
			scales=pooler_scales,
			sampling_ratio=sampling_ratio,
			pooler_type=pooler_type,
		)
		self.keypoint_head = build_keypoint_head(
			cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
		)

	def forward(
		self,
		images: ImageList,
		features: Dict[str, torch.Tensor],
		proposals: List[Instances],
		targets: Optional[List[Instances]] = None,
	) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
		"""
		See :class:`ROIHeads.forward`.
		"""
		del images
		if self.training:
			assert targets
			proposals = self.label_and_sample_proposals_mod(proposals, targets)
		del targets

		if self.training:
			losses = self._forward_keypoint(features, proposals)
			return losses
		else:
			assert proposals[0].has("pred_boxes") and proposals[0].has("pred_classes")
			pred_kps = self._forward_keypoint(features, proposals)
			return pred_kps

	def _forward_keypoint(
		self, features: Dict[str, torch.Tensor], instances: List[Instances]
	) -> Union[Dict[str, torch.Tensor], List[Instances]]:
		"""
		Forward logic of the keypoint prediction branch.
		Args:
			features (dict[str, Tensor]): mapping from feature map names to tensor.
				Same as in :meth:`ROIHeads.forward`.
			instances (list[Instances]): the per-image instances to train/predict keypoints.
				In training, they can be the proposals.
				In inference, they can be the predicted boxes.
		Returns:
			In training, a dict of losses.
			In inference, update `instances` with new fields "pred_keypoints" and return it.
		"""
		if not self.keypoint_on:
			return {} if self.training else instances

		features = [features[f] for f in self.in_features]

		if self.training:
			# The loss is defined on positive proposals with at >=1 visible keypoints.
			proposals, _ = select_foreground_proposals(instances, self.num_classes)
			proposals = select_proposals_with_visible_keypoints(proposals)
			print("proposals going to keypoint")
			print(proposals)
			proposal_boxes = [x.proposal_boxes for x in proposals]

			keypoint_features = self.keypoint_pooler(features, proposal_boxes)
			return self.keypoint_head(keypoint_features, proposals)
		else:
			pred_boxes = [x.pred_boxes for x in instances]
			keypoint_features = self.keypoint_pooler(features, pred_boxes)
			return self.keypoint_head(keypoint_features, instances)

	def _sample_proposals_mod(
		self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Based on the matching between N proposals and M groundtruth,
		sample the proposals and set their classification labels.
		Args:
			matched_idxs (Tensor): a vector of length N, each is the best-matched
				gt index in [0, M) for each proposal.
			matched_labels (Tensor): a vector of length N, the matcher's label
				(one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
			gt_classes (Tensor): a vector of length M.
		Returns:
			Tensor: a vector of indices of sampled proposals. Each is in [0, N).
			Tensor: a vector of the same length, the classification label for
				each sampled proposal. Each sample is labeled as either a category in
				[0, num_classes) or the background (num_classes).
		"""
		has_gt = gt_classes.numel() > 0
		# Get the corresponding GT for each proposal
		if has_gt:
			gt_classes = gt_classes[matched_idxs]
			# Label unmatched proposals (0 label from matcher) as background (label=num_classes)
			gt_classes[matched_labels == 0] = self.num_classes
			# Label ignore proposals (-1 label)
			gt_classes[matched_labels == -1] = -1
		else:
			gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
		N = matched_idxs.size()[0]
		sampled_idxs = torch.arange(N) 
		return sampled_idxs, gt_classes

	@torch.no_grad()
	def label_and_sample_proposals_mod(
		self, proposals: List[Instances], targets: List[Instances]
	) -> List[Instances]:
		"""
		Prepare some proposals to be used to train the ROI heads.
		It performs box matching between `proposals` and `targets`, and assigns
		training labels to the proposals.
		It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
		boxes, with a fraction of positives that is no larger than
		``self.positive_sample_fraction``.
		Args:
			See :meth:`ROIHeads.forward`
		Returns:
			list[Instances]:
				length `N` list of `Instances`s containing the proposals
				sampled for training. Each `Instances` has the following fields:
				- proposal_boxes: the proposal boxes
				- gt_boxes: the ground-truth box that the proposal is assigned to
				  (this is only meaningful if the proposal has a label > 0; if label = 0
				  then the ground-truth box is random)
				Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
		"""
		gt_boxes = [x.gt_boxes for x in targets]
		# Augment proposals with ground-truth boxes.
		# In the case of learned proposals (e.g., RPN), when training starts
		# the proposals will be low quality due to random initialization.
		# It's possible that none of these initial
		# proposals have high enough overlap with the gt objects to be used
		# as positive examples for the second stage components (box head,
		# cls head, mask head). Adding the gt boxes to the set of proposals
		# ensures that the second stage components will have some positive
		# examples from the start of training. For RPN, this augmentation improves
		# convergence and empirically improves box AP on COCO by about 0.5
		# points (under one tested configuration).
		expansion_scale = 0.05
		print("proposals before expansion")
		print(proposals)
		for i in range(len(proposals)):
			h,w = proposals[i].image_size
			prop_boxes = proposals[i].proposal_boxes.tensor
			bw = prop_boxes[:,2] - prop_boxes[:,0]
			bh = prop_boxes[:,3]-prop_boxes[:,1]
			prop_boxes[:,0] = torch.max(prop_boxes[:,0] - (bw*expansion_scale*0.5),torch.zeros_like(prop_boxes[:,0]))
			prop_boxes[:,1] = torch.max(prop_boxes[:,1] - (bh*expansion_scale*0.5),torch.zeros_like(prop_boxes[:,1]))
			prop_boxes[:,2] = torch.min(prop_boxes[:,2] + (bw*expansion_scale*0.5),torch.zeros_like(prop_boxes[:,2]) + w)
			prop_boxes[:,3] = torch.min(prop_boxes[:,3] + (bh*expansion_scale*0.5),torch.zeros_like(prop_boxes[:,3]) + h)
			proposals[i].proposal_boxes = Boxes(prop_boxes)
		print("proposals after expansion")
		print(proposals)
		if self.proposal_append_gt:
			proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

		proposals_with_gt = []

		num_fg_samples = []
		num_bg_samples = []
		for proposals_per_image, targets_per_image in zip(proposals, targets):
			has_gt = len(targets_per_image) > 0
			match_quality_matrix = pairwise_iou(
				targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
			)
			matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
			sampled_idxs, gt_classes = self._sample_proposals_mod(
				matched_idxs, matched_labels, targets_per_image.gt_classes
			)

			# Set target attributes of the sampled proposals:
			proposals_per_image = proposals_per_image[sampled_idxs]
			proposals_per_image.gt_classes = gt_classes

			# We index all the attributes of targets that start with "gt_"
			# and have not been added to proposals yet (="gt_classes").
			if has_gt:
				sampled_targets = matched_idxs[sampled_idxs]
				# NOTE: here the indexing waste some compute, because heads
				# like masks, keypoints, etc, will filter the proposals again,
				# (by foreground/background, or number of keypoints in the image, etc)
				# so we essentially index the data twice.
				for (trg_name, trg_value) in targets_per_image.get_fields().items():
					if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
						proposals_per_image.set(trg_name, trg_value[sampled_targets])
			else:
				gt_boxes = Boxes(
					targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
				)
				proposals_per_image.gt_boxes = gt_boxes

			num_bg_samples.append((gt_classes == self.num_classes).sum().item())
			num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
			proposals_with_gt.append(proposals_per_image)

		# Log the number of fg/bg samples that are selected for training ROI heads
		storage = get_event_storage()
		storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
		storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

		return proposals_with_gt