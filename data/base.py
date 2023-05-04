import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as torchvision_F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import json
import os
import random
import numpy as np
from PIL import Image
import math

def bbox_to_mask(box, H, W):
    x1, y1, x2, y2 = box
    x1, x2 = int(x1 * W), int(x2 * W)
    y1, y2 = int(y1 * H), int(y2 * H)
    ret = np.zeros([H, W])
    ret[y1:y2, x1:x2] = 1
    return ret

def is_overlap(mask1, mask2):
    if (mask1 + mask2).max() > 1:
        return True
    return False

class BaseDataset:

    def __init__(self, args):
        self.max_caplen = args.max_caplen
        self.min_word_freq = args.min_word_freq
        self.min_object_size = args.min_object_size
        self.min_objects = args.min_objects
        self.max_objects = args.max_objects

        self.include_dummies = False

        transform = [T.ToTensor()]
        self.transform = T.Compose(transform)

        # TODO
        self.resize_first = True

        self.image_size = self.image_H, self.image_W = (256, 256)

        # Required
        self.image_dir = None
        self.image_ids = None

    def __len__(self):
        return len(self.image_ids)

    def _get_tgt_image(self, filename, objects=None):
        try:
            image_path = os.path.join(self.image_dir, filename.decode('UTF-8'))
        except:
            image_path = os.path.join(self.image_dir, filename)

        # image = None
        with open(image_path, 'rb') as f:
            image = Image.open(f)
            WW, HH = image.size

            if self.resize_first:
                size = min(WW, HH)
                tgt_image = torchvision_F.resize(image, [size, size])
                # tgt_image = image.resize((size, size))

                if objects is not None:
                    for obj_idx in range(len(objects)):
                        x1, y1, x2, y2 = objects[obj_idx]['box']
                        x1 = x1 / WW
                        x2 = x2 / WW
                        y1 = y1 / HH
                        y2 = y2 / HH
                        objects[obj_idx]['box'] = [x1, y1, x2, y2]
            
            tgt_image = torchvision_F.resize(tgt_image, [self.image_H, self.image_W])
            tgt_image = np.array(tgt_image)
            if len(tgt_image.shape) == 2:
                tgt_image = np.stack([tgt_image]*3, axis=-1)
        
        if objects is not None:
            return tgt_image, objects
        return tgt_image

    def _clean_objects(self, objects, relationships=None):
        # Figure out which objects appear in relationships and which don't
        obj_idxs = set(range(len(objects)))

        obj_idxs_in_crop = []
        for i, obj_idx in enumerate(obj_idxs):
            x1, y1, x2, y2 = objects[obj_idx]['box']

            crop_box_w = x2 - x1
            crop_box_h = y2 - y1

            if crop_box_w < 0.05 or crop_box_h < 0.05:
                continue
            else:
                obj_idxs_in_crop.append(obj_idx)

        obj_idxs = list(obj_idxs_in_crop)
        if len(obj_idxs) > self.max_objects - 1:
            obj_idxs = random.sample(obj_idxs, self.max_objects)
        return obj_idxs

    def _get_object_boxes(self, objects, obj_idxs):

        if self.include_dummies:
            O = self.max_objects + 1
        else:
            O = self.max_objects
        tgt_objs = np.full([O], self.vocab.__obj_pad__)
        tgt_boxes = np.array([[0.5, 0.5, 1., 1.]]).repeat(O, 0)
        
        obj_idx_mapping = {}

        objs_count = 0
        if self.include_dummies:
            tgt_objs[objs_count] = self.vocab.__image__
            objs_count += 1
    
        for obj_idx in obj_idxs:
            obj_idx_mapping[obj_idx] = objs_count
            
            x1, y1, x2, y2 = objects[obj_idx]['box']
            obj_name = objects[obj_idx]['class']
            obj_class = self.vocab.obj_name2class[obj_name]

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            tgt_objs[objs_count] = obj_class
            tgt_boxes[objs_count] = [cx, cy, (x2 - x1), (y2 - y1)]

            objs_count += 1

        return obj_idx_mapping, tgt_objs, tgt_boxes

    def _ltrb2cxywh(self, box, in_type='float', out_type='float'):
        x1, y1, x2, y2 = box
        if in_type == 'int':
            x1, x2 = x1 / self.image_W, x2 / self.image_W
            y1, y2 = y1 / self.image_H, y2 / self.image_H
        elif in_type == 'float':
            pass
        else:
            return ValueError

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        if out_type == 'int':
            cx, w = int(cx * self.image_W), int(w * self.image_W)
            cy, h = int(cy * self.image_H), int(h * self.image_H)
            return [cx, cy, w, h]
        elif out_type == 'float':
            return [cx, cy, w, h]
        else:
            return ValueError

    def _cxywh2ltrb(self, box, in_type='float', out_type='float'):
        cx, cy, w, h = box
        if in_type == 'int':
            cx, w = cx / self.image_W, w / self.image_W
            cy, h = cy / self.image_H, h / self.image_H
        elif in_type == 'float':
            pass
        else:
            return ValueError

        x1 = cx - w/2
        x2 = cx + w/2
        y1 = cy - h/2
        y2 = cy + h/2

        if out_type == 'int':
            x1, x2 = int(x1 * self.image_W), int(x2 * self.image_W)
            y1, y2 = int(y1 * self.image_H), int(y2 * self.image_H)
            return [x1, y1, x2, y2]
        elif out_type == 'float':
            return [x1, y1, x2, y2]
        else:
            return ValueError
