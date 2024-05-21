import math
import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import torch
import torchvision.models.detection._utils as _detection_utils
from torch import Tensor, nn
from torchvision.models._api import Weights, WeightsEnum
from torchvision.models._meta import _COCO_CATEGORIES
from torchvision.models.detection.backbone_utils import (
    _resnet_fpn_extractor,
    _validate_trainable_layers,
)

# from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.resnet import ResNet50_Weights, resnet50
from torchvision.ops import boxes as box_ops
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops import sigmoid_focal_loss
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.transforms._presets import ObjectDetection

from lt_lib.architecture.anchor_utils import AnchorGenerator

__all__ = [
    "RetinaNet",
    "RetinaNet_ResNet50_FPN_Weights",
    "RetinaNet_ResNet50_FPN_V2_Weights",
    "retinanet_resnet50_fpn",
    "retinanet_resnet50_fpn_v2",
]


def _sum(x: List[Tensor]) -> Tensor:
    res = x[0]
    for i in x[1:]:
        res = res + i
    return res


def _v1_to_v2_weights(state_dict, prefix):
    for i in range(4):
        for type in ["weight", "bias"]:
            old_key = f"{prefix}conv.{2*i}.{type}"
            new_key = f"{prefix}conv.{i}.0.{type}"
            if old_key in state_dict:
                state_dict[new_key] = state_dict.pop(old_key)


def _default_anchorgen():
    anchors_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchors_sizes)
    anchor_generator = AnchorGenerator(anchors_sizes, aspect_ratios)
    return anchor_generator


class RetinaNetHead(nn.Module):
    """
    Identical class version of the RetinaNetHead class from PyTorch 2.1.2.
    A regression and classification head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
        norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
    """

    def __init__(self, in_channels, num_anchors, num_classes, norm_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__()
        self.classification_head = RetinaNetClassificationHead(
            in_channels, num_anchors, num_classes, norm_layer=norm_layer
        )
        self.regression_head = RetinaNetRegressionHead(in_channels, num_anchors, norm_layer=norm_layer)

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor], List[Tensor]) -> Dict[str, Tensor]
        return {
            "classification": self.classification_head.compute_loss(targets, head_outputs, matched_idxs),
            "bbox_regression": self.regression_head.compute_loss(targets, head_outputs, anchors, matched_idxs),
        }

    def forward(self, x):
        # type: (List[Tensor]) -> Dict[str, Tensor]
        return {"cls_logits": self.classification_head(x), "bbox_regression": self.regression_head(x)}


class RetinaNetClassificationHead(nn.Module):
    """
    Identical class version of the RetinaNetClassificationHead class from PyTorch 2.1.2.
    A classification head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
        norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
    """

    _version = 2

    def __init__(
        self,
        in_channels,
        num_anchors,
        num_classes,
        prior_probability=0.01,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()

        conv = []
        for _ in range(4):
            conv.append(misc_nn_ops.Conv2dNormActivation(in_channels, in_channels, norm_layer=norm_layer))
        self.conv = nn.Sequential(*conv)

        for layer in self.conv.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)

        self.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        torch.nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability))

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # This is to fix using _detection_utils.Matcher.BETWEEN_THRESHOLDS in TorchScript.
        # TorchScript doesn't support class attributes.
        # https://github.com/pytorch/vision/pull/1697#issuecomment-630255584
        self.BETWEEN_THRESHOLDS = _detection_utils.Matcher.BETWEEN_THRESHOLDS

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            _v1_to_v2_weights(state_dict, prefix)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def compute_loss(self, targets, head_outputs, matched_idxs):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor]) -> Tensor
        losses = []

        cls_logits = head_outputs["cls_logits"]

        for targets_per_image, cls_logits_per_image, matched_idxs_per_image in zip(targets, cls_logits, matched_idxs):
            # determine only the foreground
            foreground_idxs_per_image = matched_idxs_per_image >= 0
            num_foreground = foreground_idxs_per_image.sum()

            # create the target classification
            gt_classes_target = torch.zeros_like(cls_logits_per_image)
            gt_classes_target[
                foreground_idxs_per_image,
                targets_per_image["labels"][matched_idxs_per_image[foreground_idxs_per_image]],
            ] = 1.0

            # find indices for which anchors should be ignored
            valid_idxs_per_image = matched_idxs_per_image != self.BETWEEN_THRESHOLDS

            # compute the classification loss
            losses.append(
                sigmoid_focal_loss(
                    cls_logits_per_image[valid_idxs_per_image],
                    gt_classes_target[valid_idxs_per_image],
                    reduction="sum",
                )
                / max(1, num_foreground)
            )

        return _sum(losses) / len(targets)

    def forward(self, x):
        # type: (List[Tensor]) -> Tensor
        all_cls_logits = []

        for feature_maps in x:
            cls_logits = self.conv(feature_maps)
            cls_logits = self.cls_logits(cls_logits)

            # Permute classification output from (N, A * K, H, W) to (N, HWA, K).
            N, _, H, W = cls_logits.shape
            cls_logits = cls_logits.view(N, -1, self.num_classes, H, W)
            cls_logits = cls_logits.permute(0, 3, 4, 1, 2)
            cls_logits = cls_logits.reshape(N, -1, self.num_classes)  # Size=(N, HWA, K)

            all_cls_logits.append(cls_logits)

        return torch.cat(all_cls_logits, dim=1)


class RetinaNetRegressionHead(nn.Module):
    """
    Identical class version of the RetinaNetRegressionHead class from PyTorch 2.1.2.
    A regression head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
    """

    _version = 2

    __annotations__ = {
        "box_coder": _detection_utils.BoxCoder,
    }

    def __init__(self, in_channels, num_anchors, norm_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__()

        conv = []
        for _ in range(4):
            conv.append(misc_nn_ops.Conv2dNormActivation(in_channels, in_channels, norm_layer=norm_layer))
        self.conv = nn.Sequential(*conv)

        self.bbox_reg = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.bbox_reg.weight, std=0.01)
        torch.nn.init.zeros_(self.bbox_reg.bias)

        for layer in self.conv.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)

        self.box_coder = _detection_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self._loss_type = "l1"

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            _v1_to_v2_weights(state_dict, prefix)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor], List[Tensor]) -> Tensor
        losses = []

        bbox_regression = head_outputs["bbox_regression"]

        for targets_per_image, bbox_regression_per_image, anchors_per_image, matched_idxs_per_image in zip(
            targets, bbox_regression, anchors, matched_idxs
        ):
            # determine only the foreground indices, ignore the rest
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            num_foreground = foreground_idxs_per_image.numel()

            # select only the foreground boxes
            matched_gt_boxes_per_image = targets_per_image["boxes"][matched_idxs_per_image[foreground_idxs_per_image]]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]

            # compute the loss
            losses.append(
                _detection_utils._box_loss(
                    self._loss_type,
                    self.box_coder,
                    anchors_per_image,
                    matched_gt_boxes_per_image,
                    bbox_regression_per_image,
                )
                / max(1, num_foreground)
            )

        return _sum(losses) / max(1, len(targets))

    def forward(self, x):
        # type: (List[Tensor]) -> Tensor
        all_bbox_regression = []

        for feature_maps in x:
            bbox_regression = self.conv(feature_maps)
            bbox_regression = self.bbox_reg(bbox_regression)

            # Permute bbox regression output from (N, 4 * A, H, W) to (N, HWA, 4).
            N, _, H, W = bbox_regression.shape
            bbox_regression = bbox_regression.view(N, -1, 4, H, W)
            bbox_regression = bbox_regression.permute(0, 3, 4, 1, 2)
            bbox_regression = bbox_regression.reshape(N, -1, 4)  # Size=(N, HWA, 4)

            all_bbox_regression.append(bbox_regression)

        return torch.cat(all_bbox_regression, dim=1)


class RetinaNet(nn.Module):
    """
    Modified version of the RetinaNet class from PyTorch 2.1.2 to simplify its parametrization and return more
    information to be able to run insightful post-processing operations.
    It implements the RetinaNet.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending on if it is in training or evaluation mode.

    During training, the model expects both the input tensors and targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores for each prediction

    Args:
        backbone (nn.Module): the network used to compute the feature_maps for the model.
            It should contain an out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or an OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        head (nn.Module): Module run on top of the feature pyramid.
            Defaults to a module containing a classification and regression module.
        score_thresh (float): Score threshold used for postprocessing the detections.
        nms_iou_thresh (float): NMS threshold used for postprocessing the detections.
        nb_max_detections_per_img (int): Number of best detections to keep after NMS.
        fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training.
        bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training.
        topk_candidates (int): Number of best detections to keep before NMS.

    Example:

        >>> import torch
        >>> import torchvision
        >>> from torchvision.models.detection import RetinaNet
        >>> from torchvision.models.detection.anchor_utils import AnchorGenerator
        >>> # load a pre-trained model for classification and return
        >>> # only the feature_maps
        >>> backbone = torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).feature_maps
        >>> # RetinaNet needs to know the number of
        >>> # output channels in a backbone. For mobilenet_v2, it's 1280,
        >>> # so we need to add it here
        >>> backbone.out_channels = 1280
        >>>
        >>> # let's make the network generate 5 x 3 anchors per spatial
        >>> # location, with 5 different sizes and 3 different aspect
        >>> # ratios. We have a Tuple[Tuple[int]] because each feature
        >>> # map could potentially have different sizes and
        >>> # aspect ratios
        >>> anchor_generator = AnchorGenerator(
        >>>     sizes=((32, 64, 128, 256, 512),),
        >>>     aspect_ratios=((0.5, 1.0, 2.0),)
        >>> )
        >>>
        >>> # put the pieces together inside a RetinaNet model
        >>> model = RetinaNet(backbone,
        >>>                   num_classes=2,
        >>>                   anchor_generator=anchor_generator)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
    """

    __annotations__ = {
        "box_coder": _detection_utils.BoxCoder,
        "proposal_matcher": _detection_utils.Matcher,
    }

    def __init__(
        self,
        backbone,
        num_classes,
        # Anchor parameters
        anchor_generator=None,
        head=None,
        proposal_matcher=None,
        fg_iou_thresh=0.5,
        bg_iou_thresh=0.4,
        # Process head-outputs
        process_detections_during_training=True,
        score_thresh=0.1,
        nms_iou_thresh=0.7,
        topk_candidates=500,
        nb_max_detections_per_img=300,
        **kwargs,
    ):
        super().__init__()
        # _log_api_usage_once(self)
        self.num_classes = num_classes

        # Check for attribute 'out_channels' in arg 'backbone'
        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )
        self.backbone = backbone

        # Assign anchor generator
        if not isinstance(anchor_generator, (AnchorGenerator, type(None))):
            raise TypeError(
                f"anchor_generator should be of type AnchorGenerator or None instead of {type(anchor_generator)}"
            )
        if anchor_generator is None:
            anchor_generator = _default_anchorgen()
        self.anchor_generator = anchor_generator

        # Instantiate RetinaNet Head
        if head is None:
            head = RetinaNetHead(
                backbone.out_channels, anchor_generator.num_anchors_per_location_per_pyramid_level()[0], num_classes
            )
        self.head = head

        # Assigns the matcher between GTs and anchors
        if proposal_matcher is None:
            proposal_matcher = _detection_utils.Matcher(
                fg_iou_thresh,
                bg_iou_thresh,
                allow_low_quality_matches=True,
            )
        self.proposal_matcher = proposal_matcher

        # if image_mean is None:
        #     image_mean = [0.485, 0.456, 0.406]
        # if image_std is None:
        #     image_std = [0.229, 0.224, 0.225]
        # self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std, **kwargs)

        # Proces head-outputs parameters
        self.process_detections_during_training = process_detections_during_training
        self.score_thresh = score_thresh
        self.nms_iou_thresh = nms_iou_thresh
        self.topk_candidates = topk_candidates
        self.nb_max_detections_per_img = nb_max_detections_per_img

        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def compute_loss(self, targets, head_outputs, anchors):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor]) -> Dict[str, Tensor]
        matched_idxs = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            if targets_per_image["boxes"].numel() == 0:
                matched_idxs.append(
                    torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device)
                )
                continue

            match_quality_matrix = box_ops.box_iou(targets_per_image["boxes"], anchors_per_image)
            matched_idxs.append(self.proposal_matcher(match_quality_matrix))

        return self.head.compute_loss(targets, head_outputs, anchors, matched_idxs)

    def forward(
        self, images: list[Tensor], targets: list[Dict[str, Tensor]] | None = None
    ) -> tuple[dict[str, Tensor], list[dict[str, Tensor]]]:
        """
        Modified forward pass of the RetinaNet model.

        Args:
            images: images to be processed
            targets: ground-truth boxes present in the image (optional)

        Returns:
            losses and detections: the output from the model.
        """
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be None when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    torch._assert(isinstance(boxes, torch.Tensor), "Expected target boxes to be of type Tensor.")
                    torch._assert(
                        len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                        "Expected target boxes to be a tensor of shape [N, 4].",
                    )

        # get the original image sizes
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        # images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        check_for_degenerate_bboxes(targets)

        # get the feature_maps from the backbone
        feature_maps = self.backbone(images)
        # if isinstance(feature_maps, torch.Tensor):
        #     feature_maps = OrderedDict([("0", feature_maps)])

        # TODO: Do we want a list or a dict?
        feature_maps = list(feature_maps.values())
        n_anchors_per_location = self.anchor_generator.num_anchors_per_location_per_pyramid_level()[0]
        num_anchors_per_level = [n_anchors_per_location * f_map.size(2) * f_map.size(3) for f_map in feature_maps]

        # compute the retinanet heads outputs using the feature_maps
        head_outputs = self.head(feature_maps)

        # create the set of anchors
        anchors = self.anchor_generator(images, feature_maps)

        losses = {}
        detections = []

        if targets is not None:
            # compute the losses
            losses = self.compute_loss(targets, head_outputs, anchors)

        if torch.jit.is_scripting() and self.process_detections_during_training and not self._has_warned:
            warnings.warn("This custom implementation of the RetinaNet always returns a tuple: (Losses, Detctions).")
            self._has_warned = True

        # If model is in eval mode or if param process_detections_during_training is at True, processes detections
        if not self.training or self.process_detections_during_training:
            detections = self.process_model_head_outputs(
                head_outputs=head_outputs,
                anchors=anchors,
                image_shapes=original_image_sizes,
                num_anchors_per_level=num_anchors_per_level,
                box_coder=_detection_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0)),
            )

        return losses, detections

    def process_model_head_outputs(
        self,
        head_outputs,
        anchors,
        image_shapes,
        num_anchors_per_level,
        box_coder,
    ):
        """
        Splits and processes the model head outputs to generate detections with increased information compared to the
        original processing funciton from PyTorch 2.1.2.

        Args:
            head_outputs: Dictionary containing tensors of class logits and bounding box regression outputs.
            anchors: List of tensors containing anchor boxes.
            image_shapes: List of tuples representing the shapes of input images.
            num_anchors_per_level: Number of anchors per pyramid level.
            box_coder: Box coder object for decoding bounding box predictions.

        Returns:
            List[Dict[str, Tensor]]: A list of dictionaries containing detected boxes, scores, and probable labels for
                each image.

        """
        # type : (Dict[str, List[Tensor]], List[List[Tensor]], List[Tuple[int, int]]) -> List[Dict[str, Tensor]]

        # Splits outputs per pyramid level 'cls_logits' and 'bbox_regression'
        split_head_outputs: Dict[str, List[Tensor]] = {}
        # head_outputs has 2 attributes:
        for k in head_outputs:
            split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))

        # Splits anchors per pyramid level
        split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

        class_logits = split_head_outputs["cls_logits"]
        box_regression = split_head_outputs["bbox_regression"]

        num_images = len(image_shapes)

        detections: List[Dict[str, Tensor]] = []

        # Processes images 1 by 1
        for index in range(num_images):
            # Extracts the image bboxes regressed and class labels associated from the tensor
            box_regression_per_image = [br[index] for br in box_regression]
            logits_per_image = [cl[index] for cl in class_logits]
            anchors_per_image, image_shape = split_anchors[index], image_shapes[index]

            image_boxes = []
            image_scores = []
            image_scores_max = []
            image_most_probable_classes = []

            # Processes levels 1 by 1
            for box_regression_per_level, logits_per_level, anchors_per_level in zip(
                box_regression_per_image, logits_per_image, anchors_per_image
            ):
                # Removes low scoring boxes
                scores_per_level = torch.sigmoid(logits_per_level)
                scores_per_level = scores_per_level[:, 1:]  # Removes background class
                scores_max_per_level_values, scores_max_per_level_indices = scores_per_level.max(dim=1)
                keep_mask = scores_max_per_level_values > self.score_thresh
                scores_max_per_level_values = scores_max_per_level_values[keep_mask]
                scores_max_per_level_indices = scores_max_per_level_indices[keep_mask]
                topk_idxs = torch.where(keep_mask)[0]

                # Keep only topk scoring predictions
                num_topk = _detection_utils._topk_min(topk_idxs, self.topk_candidates, 0)
                scores_max_per_level_values, idxs = scores_max_per_level_values.topk(num_topk)
                scores_max_per_level_indices = scores_max_per_level_indices[idxs]
                topk_idxs = topk_idxs[idxs]

                # Decodes the bboxes positions on the image and clip it to the image
                boxes_per_level = box_coder.decode_single(
                    box_regression_per_level[topk_idxs], anchors_per_level[topk_idxs]
                )
                boxes_per_level = box_ops.clip_boxes_to_image(boxes_per_level, image_shape)

                # Get best anchors scores for all classes
                scores_per_level = scores_per_level[topk_idxs]

                image_boxes.append(boxes_per_level)
                image_scores.append(scores_per_level)
                image_scores_max.append(scores_max_per_level_values)
                image_most_probable_classes.append(scores_max_per_level_indices)

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_scores_max = torch.cat(image_scores_max, dim=0)
            image_most_probable_classes = torch.cat(image_most_probable_classes, dim=0)

            # non-maximum suppression
            keep = box_ops.batched_nms(image_boxes, image_scores_max, image_most_probable_classes, self.nms_iou_thresh)
            keep = keep[: self.nb_max_detections_per_img]

            detections.append(
                {
                    "boxes": image_boxes[keep],
                    "scores": image_scores[keep],
                    "probable_labels": image_most_probable_classes[keep] + 1,  # Because background column was removed
                }
            )

        return detections


def check_for_degenerate_bboxes(targets):
    # Check for degenerate boxes
    if targets is not None:
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenerate box
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                torch._assert(
                    False,
                    "All bounding boxes should have positive height and width."
                    f" Found invalid box {degen_bb} for target at index {target_idx}.",
                )


_COMMON_META = {
    "categories": _COCO_CATEGORIES,
    "min_size": (1, 1),
}


class RetinaNet_ResNet50_FPN_Weights(WeightsEnum):
    COCO_V1 = Weights(
        url="https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth",
        transforms=ObjectDetection,
        meta={
            **_COMMON_META,
            "num_params": 34014999,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/detection#retinanet",
            "_metrics": {
                "COCO-val2017": {
                    "box_map": 36.4,
                }
            },
            "_ops": 151.54,
            "_file_size": 130.267,
            "_docs": """These weights were produced by following a similar training recipe as on the paper.""",
        },
    )
    DEFAULT = COCO_V1


class RetinaNet_ResNet50_FPN_V2_Weights(WeightsEnum):
    COCO_V1 = Weights(
        url="https://download.pytorch.org/models/retinanet_resnet50_fpn_v2_coco-5905b1c5.pth",
        transforms=ObjectDetection,
        meta={
            **_COMMON_META,
            "num_params": 38198935,
            "recipe": "https://github.com/pytorch/vision/pull/5756",
            "_metrics": {
                "COCO-val2017": {
                    "box_map": 41.5,
                }
            },
            "_ops": 152.238,
            "_file_size": 146.037,
            "_docs": """These weights were produced using an enhanced training recipe to boost the model accuracy.""",
        },
    )
    DEFAULT = COCO_V1


def retinanet_with_resnet50(
    *,
    num_classes: int,
    anchors_sizes: tuple[int] = (16, 32, 64, 128, 256),
    anchors_scales: tuple[float] | tuple[tuple[float]] = (1.0, 1.33, 1.66),
    anchors_ratios: tuple[float] = (0.5, 1.0, 2.0),
    weights: Optional[RetinaNet_ResNet50_FPN_Weights] = None,
    progress: bool = True,
    weights_backbone: Optional[ResNet50_Weights] = ResNet50_Weights.IMAGENET1K_V1,
    trainable_backbone_layers: Optional[int] = None,
    norm_layer: Literal["BatchNorm2d", "FrozenBatchNorm2d"],
    **kwargs: Any,
) -> RetinaNet:
    """Creates a RetinaNet model with a ResNet50 backbone.

    Args:
        num_classes: Number of classes for classification.
        anchors_sizes: Anchor sizes for feature maps.
        anchors_scales: Anchor scales for feature maps.
        anchors_ratios: Anchor aspect ratios for feature maps.
        weights: Weights for the model. Defaults to None.
        progress: If True, displays a progress bar of the download. Defaults to True.
        weights_backbone: Backbone weights for the ResNet50 model. Defaults to "ResNet50_Weights.IMAGENET1K_V1".
        trainable_backbone_layers: Number of trainable layers in the backbone. Defaults to None.
        norm_layer: Type of normalization layer to use.
        **kwargs: Additional keyword arguments passed to the RetinaNet constructor.

    Returns:
        RetinaNet: A RetinaNet model with a ResNet50 backbone.
    """
    # weights = RetinaNet_ResNet50_FPN_Weights.verify(weights)
    weights_backbone = ResNet50_Weights.verify(weights_backbone)

    if num_classes is None:
        num_classes = 91

    is_trained = weights is not None or weights_backbone is not None
    trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)
    norm_layer = misc_nn_ops.FrozenBatchNorm2d if norm_layer == "FrozenBatchNorm2d" else nn.BatchNorm2d

    # Instantiate the ResNet50 backbone
    backbone = resnet50(weights=weights_backbone, progress=progress, norm_layer=norm_layer)
    # Modify the backbone so that it returns the feature_maps of the FPN
    # skip P2 because it generates too many anchors (according to the Focal Loss aka RetinaNet paper)
    backbone = _resnet_fpn_extractor(
        backbone, trainable_backbone_layers, returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(256, 256)
    )

    anchor_generator = AnchorGenerator(anchors_sizes, anchors_scales, anchors_ratios)

    model = RetinaNet(backbone, num_classes, anchor_generator=anchor_generator, **kwargs)

    if weights is not None:
        model.load_state_dict(
            weights.get_state_dict(progress=progress, check_hash=True)
            if isinstance(weights, RetinaNet_ResNet50_FPN_Weights)
            else weights
        )
        if weights == RetinaNet_ResNet50_FPN_Weights.COCO_V1:
            _detection_utils.overwrite_eps(model, 0.0)

    return model
