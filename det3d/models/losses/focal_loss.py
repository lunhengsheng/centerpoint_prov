import torch.nn as nn
import torch.nn.functional as F
from det3d.ops.sigmoid_focal_loss import sigmoid_focal_loss as _sigmoid_focal_loss

from ..registry import LOSSES
from .utils import weight_reduce_loss

from abc import ABCMeta, abstractmethod
import numpy as np
import torch



def indices_to_dense_vector(indices,
                            size,
                            indices_value=1.,
                            default_value=0,
                            dtype=np.float32):
  """Creates dense vector with indices set to specific value and rest to zeros.
  This function exists because it is unclear if it is safe to use
    tf.sparse_to_dense(indices, [size], 1, validate_indices=False)
  with indices which are not ordered.
  This function accepts a dynamic size (e.g. tf.shape(tensor)[0])
  Args:
    indices: 1d Tensor with integer indices which are to be set to
        indices_values.
    size: scalar with size (integer) of output Tensor.
    indices_value: values of elements specified by indices in the output vector
    default_value: values of other elements in the output vector.
    dtype: data type.
  Returns:
    dense 1D Tensor of shape [size] with indices set to indices_values and the
        rest set to default_value.
  """
  dense = torch.zeros(size).fill_(default_value)
  dense[indices] = indices_value

  return dense


# This method is only for debugging
def py_sigmoid_focal_loss(
    pred, target, weight=None, gamma=2.0, alpha=0.25, reduction="mean", avg_factor=None
):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = (
        F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        * focal_weight
    )
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def sigmoid_focal_loss(
    pred, target, weight=None, gamma=2.0, alpha=0.25, reduction="mean", avg_factor=None
):
    # Function.apply does not accept keyword arguments, so the decorator
    # "weighted_loss" is not applicable
    loss = _sigmoid_focal_loss(pred, target, gamma, alpha)
    # TODO: find a proper way to handle the shape of weight
    if weight is not None:
        weight = weight.view(-1, 1)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module
class FocalLoss(nn.Module):
    def __init__(
        self, use_sigmoid=True, gamma=2.0, alpha=0.25, reduction="mean", loss_weight=1.0
    ):
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, "Only sigmoid focal loss supported now."
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self, pred, target, weight=None, avg_factor=None, reduction_override=None
    ):
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        if self.use_sigmoid:
            loss_cls = self.loss_weight * sigmoid_focal_loss(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor,
            )
        else:
            raise NotImplementedError
        return loss_cls

class Loss(object):
  """Abstract base class for loss functions."""
  __metaclass__ = ABCMeta

  def __call__(self,
               prediction_tensor,
               target_tensor,
               ignore_nan_targets=False,
               scope=None,
               **params):
    """Call the loss function.
    Args:
      prediction_tensor: an N-d tensor of shape [batch, anchors, ...]
        representing predicted quantities.
      target_tensor: an N-d tensor of shape [batch, anchors, ...] representing
        regression or classification targets.
      ignore_nan_targets: whether to ignore nan targets in the loss computation.
        E.g. can be used if the target tensor is missing groundtruth data that
        shouldn't be factored into the loss.
      scope: Op scope name. Defaults to 'Loss' if None.
      **params: Additional keyword arguments for specific implementations of
              the Loss.
    Returns:
      loss: a tensor representing the value of the loss function.
    """
    if ignore_nan_targets:
      target_tensor = torch.where(torch.isnan(target_tensor),
                                prediction_tensor,
                                target_tensor)
    return self._compute_loss(prediction_tensor, target_tensor, **params)


def _sigmoid_cross_entropy_with_logits(logits, labels):
  # to be compatible with tensorflow, we don't use ignore_idx
  loss = torch.clamp(logits, min=0) - logits * labels.type_as(logits)
  loss += torch.log1p(torch.exp(-torch.abs(logits)))
  # loss = nn.BCEWithLogitsLoss(reduce="none")(logits, labels.type_as(logits))
  # transpose_param = [0] + [param[-1]] + param[1:-1]
  # logits = logits.permute(*transpose_param)
  # loss_ftor = nn.NLLLoss(reduce=False)
  # loss = loss_ftor(F.logsigmoid(logits), labels)
  return loss


class SigmoidFocalClassificationLoss(Loss):
  """Sigmoid focal cross entropy loss.
  Focal loss down-weights well classified examples and focusses on the hard
  examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.
  """

  def __init__(self, gamma=2.0, alpha=0.25):
    """Constructor.
    Args:
      gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
      alpha: optional alpha weighting factor to balance positives vs negatives.
      all_zero_negative: bool. if True, will treat all zero as background.
        else, will treat first label as background. only affect alpha.
    """
    self._alpha = alpha
    self._gamma = gamma

  def _compute_loss(self,
                    prediction_tensor,
                    target_tensor,
                    weights,
                    class_indices=None):
    """Compute loss function.
    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
      target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
      weights: a float tensor of shape [batch_size, num_anchors]
      class_indices: (Optional) A 1-D integer tensor of class indices.
        If provided, computes loss only for the specified class indices.
    Returns:
      loss: a float tensor of shape [batch_size, num_anchors, num_classes]
        representing the value of the loss function.
    """
    weights = weights.unsqueeze(2)
    if class_indices is not None:
      weights *= indices_to_dense_vector(class_indices,
            prediction_tensor.shape[2]).view(1, 1, -1).type_as(prediction_tensor)
    per_entry_cross_ent = (_sigmoid_cross_entropy_with_logits(
        labels=target_tensor, logits=prediction_tensor))
    prediction_probabilities = torch.sigmoid(prediction_tensor)
    p_t = ((target_tensor * prediction_probabilities) +
           ((1 - target_tensor) * (1 - prediction_probabilities)))
    modulating_factor = 1.0
    if self._gamma:
      modulating_factor = torch.pow(1.0 - p_t, self._gamma)
    alpha_weight_factor = 1.0
    if self._alpha is not None:
      alpha_weight_factor = (target_tensor * self._alpha +
                              (1 - target_tensor) * (1 - self._alpha))

    focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor *
                                per_entry_cross_ent)
    return focal_cross_entropy_loss * weights

