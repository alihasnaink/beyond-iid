from __future__ import annotations

import torch
import torch.nn.functional as F

from methods.manifold_mixup import mix_different_classes


def _classifier_placeholder_loss(known_logits, dummy_logits, labels, beta=1.0):
    # PROSER collapses the dummy bank to its strongest response. This matches
    # the paper/reference implementation and prevents five dummy terms from
    # changing the softmax denominator merely because five placeholders exist.
    strongest_dummy = dummy_logits.max(dim=1, keepdim=True).values
    open_logits = torch.cat([known_logits, strongest_dummy], dim=1)
    known_classification = F.cross_entropy(open_logits, labels)

    masked_known = known_logits.clone()
    masked_known.scatter_(1, labels[:, None], float("-inf"))
    competitor_logits = torch.cat([masked_known, strongest_dummy], dim=1)
    dummy_target = torch.full_like(labels, known_logits.shape[1])
    boundary_loss = F.cross_entropy(competitor_logits, dummy_target)
    return known_classification + beta * boundary_loss, known_classification, boundary_loss


def _data_placeholder_loss(model, images, labels, alpha=2.0):
    layer2 = model.forward_to_layer2(images)
    mixed, permutation, lambdas = mix_different_classes(layer2, labels, alpha)
    mixed_features = model.features_from_layer2(mixed)
    known_logits, dummy_logits = model.logits_from_features(mixed_features)
    strongest_dummy = dummy_logits.max(dim=1, keepdim=True).values
    open_logits = torch.cat([known_logits, strongest_dummy], dim=1)
    dummy_target = torch.full_like(labels, known_logits.shape[1])
    return F.cross_entropy(open_logits, dummy_target), permutation, lambdas


def proser_loss(model, images, labels, beta=1.0, gamma=0.1, mixup_alpha=2.0):
    half = len(images) // 2
    if half == 0 or len(images) - half == 0:
        raise ValueError("PROSER requires both half-batches to be non-empty.")
    classifier_images, classifier_labels = images[:half], labels[:half]
    mixup_images, mixup_labels = images[half:], labels[half:]

    known_logits, dummy_logits = model(classifier_images, return_dummy=True)
    classifier_loss, known_ce, boundary_loss = _classifier_placeholder_loss(
        known_logits, dummy_logits, classifier_labels, beta,
    )
    data_loss, permutation, lambdas = _data_placeholder_loss(model, mixup_images, mixup_labels, mixup_alpha)
    total = classifier_loss + gamma * data_loss
    return total, {
        "known_ce": known_ce.detach(), "classifier_boundary_loss": boundary_loss.detach(),
        "data_placeholder_loss": data_loss.detach(), "mean_mix_lambda": lambdas.mean().detach(),
        "different_class_pairs": (mixup_labels != mixup_labels[permutation]).float().mean().detach(),
    }
