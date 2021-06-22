import torch

def accuarcy(preds, gts, ignore_index=None, average='weighted'):
    class_ids = torch.unique(gts)
    valid_mask = torch.ones_like(gts, dtype=torch.bool)
    if ignore_index!=None:
        valid_mask = gts!=ignore_index
        class_ids = class_ids[class_ids!=ignore_index]
    valid_pixels = float(torch.sum(valid_mask)) ## tensor can't be divided by int
    if average=='micro':
        return torch.sum((preds==gts)*valid_mask)/valid_pixels
    num_classes = len(class_ids)
    acc = 0.
    for class_id in class_ids:
        this_class_mask = gts==class_id
        this_class_pixels = float(torch.sum(this_class_mask))
        if average=='macro':
            acc += (torch.sum(preds==class_id)/this_class_pixels)/num_classes
        elif average=='weighted':
            acc += torch.sum(preds==class_id)/valid_pixels
    return acc


def mIoU(preds, gts, ignore_index=None, average='weighted'):
    class_ids = torch.unique(gts)
    valid_mask = torch.ones_like(gts, dtype=torch.bool)
    if ignore_index!=None:
        valid_mask = gts!=ignore_index
        class_ids = class_ids[class_ids!=ignore_index]
    valid_pixels = float(torch.sum(valid_mask)) ## tensor can't be divided by int
    _mIoU = 0.
    for class_id in class_ids:
        this_class_pred = preds==class_id
        this_class_gt = gts==class_id
        inters = torch.sum(this_class_pred & this_class_gt)
        unions = torch.sum(this_class_pred | this_class_gt)
        this_class_iou_weighted = (inters/float(unions))*(torch.sum(this_class_gt)/valid_pixels)
        _mIoU += this_class_iou_weighted
    return _mIoU