import os
import json
import copy
import collections

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np

from data.base import BaseDataset
from data.vocab import BaseVocab
from collections import Counter

import random

class CocoDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.num_workers = args.loader_num_workers
        self.dataroot = args.dataroot

        # self.vocab = CocoVocab(
        #     os.path.join(self.dataroot, 'dataset_coco.json')
        # )
        
        self.train_dataset = CocoDataset(
            self.args, split='train',
            image_dir=os.path.join(self.dataroot, "train2014"),
            instances_json_path=os.path.join(self.dataroot, "annotations/instances_train2014.json"),
            caption_json_path=os.path.join(self.dataroot, 'dataset_coco.json'),
            vocab_json_path=os.path.join(args.dataroot, "vocab.json"),
            wordmap_json_path=os.path.join(args.dataroot, "wordmap.json")
        )
        self.vocab = self.train_dataset.vocab

        # self.valid_dataset = COCO_Dataset(
        #     self.args, mode='valid',
        #     image_dir=os.path.join(self.dataroot, "images/val2017"),
        #     instances_json_path=os.path.join(self.dataroot, "annotations/instances_val2017.json"),
        #     stuff_json_path=os.path.join(self.dataroot, "annotations/stuff_val2017.json"),
        #     vocab_json_path = os.path.join(args.dataroot, "vocab.json")
        # )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, shuffle=True, 
            num_workers=self.num_workers)
    
    # def val_dataloader(self):
    #     return DataLoader(
    #         self.valid_dataset, 
    #         batch_size=self.batch_size, shuffle=False, 
    #         num_workers=self.num_workers)

class CocoVocab(BaseVocab):
    def __init__(self, fname=None, instances_data=None):
        super().__init__()

        self.save_attributes += ['coco_id_to_object_idx']
        # self.synonyms = {
        #     'person': ['man', 'woman', 'girl', 'boy']
        # }

        self.synonyms = {
            'man':'person',
            'woman': 'person',
            'girl': 'person',
            'boy': 'person'
        }

        if os.path.exists(fname):
            self.load(fname)
            self.coco_id_to_object_idx = {
                int(k): v for k, v in self.coco_id_to_object_idx.items()}
        else:
            # Add tokens
            self._add_base_tokens()

            self.coco_id_to_object_idx = {}
            # COCO label starts at 1
            for category_data in instances_data['categories']:
                category_id = category_data['id'] - 1 + len(self.obj_tokens) #starts at 5
                category_name = category_data['name']

                self.coco_id_to_object_idx[category_data['id']] = category_id                 
                self.obj_name2class[category_name] = category_id

            obj_keys = self.obj_name2class.values()
            max_obj_key = max(obj_keys)
            count = 0
            for i in range(max_obj_key):
                if i not in obj_keys:
                    count += 1
                    self.obj_name2class[f'__obj_undef_{count}__'] = i
        
            self._random_color()
            self.save(fname)
        self._add_token_attr()
        self._build_class2name()

class CocoDataset(BaseDataset):
    def __init__(self, 
        args, split, 
        image_dir, 
        instances_json_path,
        caption_json_path, 
        vocab_json_path,
        wordmap_json_path,
    ):
        super().__init__(args)
        self.image_dir = image_dir
        
        with open(instances_json_path, 'r') as f:
            instances_data = json.load(f)

        word_freq = Counter()
        with open(caption_json_path, 'r') as f:
            caption_data = json.load(f)['images']

        if not os.path.exists(wordmap_json_path):
            for x in caption_data:
                filename = x['filename']
                captions = []
                for sent in x['sentences']:
                    if len(sent['tokens']) <= self.max_caplen:
                        word_freq.update(sent['tokens'])
                        captions.append(sent['tokens'])
                # if len(captions) == 0:
                    # continue
            words = [w for w in word_freq.keys() if word_freq[w] > self.min_word_freq]
            word_map = {k: v + 1 for v, k in enumerate(words)}
            word_map['<unk>'] = len(word_map) + 1
            word_map['<start>'] = len(word_map) + 1
            word_map['<end>'] = len(word_map) + 1
            word_map['<pad>'] = 0

            with open(wordmap_json_path, 'w') as f:
                json.dump(word_map, f)
        else:
            with open(wordmap_json_path, 'r') as f:
                word_map = json.load(f)
        self.word_map = word_map

        self.id2word = {}
        for k, v in word_map.items():
            self.id2word[v] = k
        # print(self.id2word)



        self.image_filename_to_caps = {}
        for x in caption_data:
            filename = x['filename']
            self.image_filename_to_caps[filename] = []
            
            if len(x['sentences']) == 0:
                caption_enc = [word_map['<start>']] \
                        + [word_map['<end>']]
                self.image_filename_to_caps[filename].append(caption_enc)
            else:
                for sent in x['sentences']:
                    tokens = sent['tokens']
                    # print(caption)                
                    caption_enc = [word_map['<start>']] \
                            + [word_map.get(word, word_map['<unk>']) for word in tokens] \
                            + [word_map['<end>']]
                    self.image_filename_to_caps[filename].append(caption_enc)

        self.vocab = CocoVocab(
            fname=vocab_json_path,
            instances_data=instances_data
        )

        self.word2objclass = {}
        for word in self.word_map:
            if word in self.vocab.obj_name2class:
                self.word2objclass[word] = self.vocab.obj_name2class[word]
                continue
            if word in self.vocab.synonyms and self.vocab.synonyms[word] in self.vocab.obj_name2class:
                self.word2objclass[word] = self.vocab.obj_name2class[self.vocab.synonyms[word]]
                continue
        
            # print(obj_name, j)
            # if j is not None:
                # self.objname2wordid[obj_name] = j
        # print(self.word2objclass)

        self.image_ids = []
        self.image_id_to_filename = {}
        self.image_id_to_size = {}
        self.image_id_to_caps = {}
        self.image_not_found = []
        for image_data in instances_data['images']:
            image_id = image_data['id']
            filename = image_data['file_name']

            # TODO
            if not os.path.exists(os.path.join(self.image_dir, filename)):
                self.image_not_found.append(image_id)
                continue

            width = image_data['width']
            height = image_data['height']
            self.image_ids.append(image_id)
            self.image_id_to_filename[image_id] = filename
            self.image_id_to_size[image_id] = (width, height)
            self.image_id_to_caps[image_id] = self.image_filename_to_caps[filename]
        
        # Add object data from instances
        self.image_id_to_objects = collections.defaultdict(list)
        for object_data in instances_data['annotations']:
            image_id = object_data['image_id']
            x1, y1, w, h = object_data['bbox']
            try:
                W, H = self.image_id_to_size[image_id]
            except:
                continue
            box_area = (w * h) / (W * H)
            box_ok = box_area > self.min_object_size
            obj_class = self.vocab.coco_id_to_object_idx[object_data['category_id']]
            object_name = self.vocab.obj_class2name[obj_class]
            #category_ok = object_name in category_whitelist
            other_ok = object_name != 'other' or include_other
            if box_ok and other_ok:
                # mask = seg_to_mask(object_data['segmentation'], W, H)
                # box = [int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2)]
                box = [int(x1), int(y1), int(x1+w), int(y1+h)]
                # print(box)
                self.image_id_to_objects[image_id].append({
                    'box': box,
                    'class': object_name,
                    # 'segmentation': mask,
                })

        # sta_dict_path = os.path.dirname(instances_json_path)
        # with open(os.path.join(sta_dict_path,'sta_dict.json'), 'w') as fp:
        #     json.dump(sta_dict, fp)

        # Prune images that have too few or too many objects
        new_image_ids = []
        total_objs = 0
        for image_id in self.image_ids:
            num_objs = len(self.image_id_to_objects[image_id])
            total_objs += num_objs
            if self.min_objects <= num_objs:
                new_image_ids.append(image_id)
        self.image_ids = new_image_ids

    # @profile
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        filename = self.image_id_to_filename[image_id]
        caption = random.choice(self.image_id_to_caps[image_id])

        objects = copy.deepcopy(self.image_id_to_objects[image_id])
        

        tgt_image, objects = self._get_tgt_image(filename, objects)
        
        obj_idxs = self._clean_objects(objects)
        
        obj_idx_mapping, tgt_objs, tgt_boxes = self._get_object_boxes(objects, obj_idxs)
        # bbox_masks = np.zeros([len(caption), 1, self.image_H, self.image_W])

        has_mask = np.zeros(len(caption))
        # for i, wordid in enumerate(caption):
        #     obj_class = self.word2objclass.get(wordid, None)
        #     if obj_class in tgt_objs:
        #         idx = tgt_objs.index(obj_class)
        #         x1, y1, x2, y2 = self._cxywh2ltrb(inputs['tgt_boxes'][idx], 'float', 'int')
        #         bbox_masks[i][0, y1:y2, x1:x2] = 1.
        #         has_mask[i] = 1.
        
        tgt_image = self.transform(tgt_image)

        inputs = {}
        inputs['tgt_image'] = 2*tgt_image.float() - 1
        # inputs['tgt_boxes'] = torch.FloatTensor(tgt_boxes)
        # inputs['tgt_objs'] = torch.LongTensor(tgt_objs).unsqueeze(-1)
        # inputs['tgt_masks'] = torch.FloatTensor(bbox_masks)
        inputs['has_mask'] = torch.FloatTensor(has_mask)
        inputs['caption'] = torch.LongTensor(caption)
        inputs['caplen'] = torch.LongTensor([len(caption)])

        inputs['image_id'] = torch.LongTensor([image_id])
        print('get')

        return inputs

    def collate_fn(self, inputs):
        """
        Collate function to be used when wrapping a CLEVRDialogDataset in a
        DataLoader. Returns a tuple of the following:

        - imgs: FloatTensor of shape (N, C, H, W)
        - objs: LongTensor of shape (O,) giving categories for all objects
        - boxes: FloatTensor of shape (O, 4) giving boxes for all objects
        - triplets: FloatTensor of shape (T, 3) giving all triplets, where
        triplets[t] = [i, p, j] means that [objs[i], p, objs[j]] is a triple
        - obj_to_img: LongTensor of shape (O,) mapping objects to images;
        obj_to_img[i] = n means that objs[i] belongs to imgs[n]
        - triple_to_img: LongTensor of shape (T,) mapping triplets to images;
        triple_to_img[t] = n means that triplets[t] belongs to imgs[n].
        """
        # inputs is a list, and each element is (image, objs, boxes, triplets)
        #all_imgs, all_boxes, all_triplets, all_conv_counts, all_triplet_type = [], [], [], [], []
        print('collate')
        ret = {}
        for key in inputs[0].keys():
            ret[key] = []
    
        # max_objects = 0
        max_caplen = 0

        for _, it in enumerate(inputs):
            L = it['caption'].shape[0]
            # print(L)
            max_caplen = max(max_caplen, L)

        #print(f'O = {max_objects}, T = {max_triplets}')

        for b, it in enumerate(inputs):
            print(b)
            ret['image_id'].append(it['image_id'])
            ret['tgt_image'].append(it['tgt_image'])
            ret['caplen'].append(it['caplen'])
            
            # Padded objs
            L = it['caplen']
            if max_caplen - L > 0:
                zeros_v = torch.full((max_caplen - L ,), self.word_map['<pad>'], dtype=torch.long)
                ret['caption'].append(torch.cat([it['caption'], zeros_v]))

                # zeros_m = torch.full((max_caplen - L, 1 , self.image_H, self.image_W), 0, dtype=torch.long)
                # ret['tgt_masks'].append(torch.cat([it['tgt_masks'], zeros_m]))

                zeros_f = torch.full((max_caplen - L ,), 0., dtype=torch.float)
                ret['has_mask'].append(torch.cat([it['has_mask'], zeros_f]))
            else:
                ret['caption'].append(it['caption'])
                # ret['tgt_masks'].append(it['tgt_masks'])
                ret['has_mask'].append(it['has_mask'])
            

        for key, value in ret.items():
            # print(key, value)
            ret[key] = torch.stack(value, dim=0)


        return ret
