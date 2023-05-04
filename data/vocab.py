import json
import torch
import numpy as np

class BaseVocab:
    def __init__(self):
        self.obj_tokens = [
            '__obj_pad__',
            '__image__',
        ]
        self.undef_tokens = []

        self.obj_name2class = {}
        self.obj_name2color = {}
        self.save_attributes = ['obj_name2class', 'obj_name2color']

    def _random_color(self):
        colors = np.random.randint(0, 256, size=[len(self.obj_name2class), 3])
        for obj_name, obj_class in self.obj_name2class.items():
            self.obj_name2color[obj_name] = tuple(colors[obj_class].tolist())
        
    def _add_base_tokens(self):
        for token in self.obj_tokens:
            class_idx = len(self.obj_name2class)
            setattr(self, token, class_idx)
            self.obj_name2class[token] = class_idx
    
    def _add_undef_classes(self):
        obj_keys = self.obj_name2class.values()
        max_obj_key = max(obj_keys)
        count = 0
        for i in range(max_obj_key):
            if i not in obj_keys:
                count += 1
                undef_name = f'__obj_undef_{count}__'
                self.obj_name2class[undef_name] = i
                self.undef_tokens.append(undef_name)

    def _add_token_attr(self, location=False):
        for token in self.obj_tokens:
            class_idx = self.obj_name2class[token]
            setattr(self, token, class_idx)

    def _build_class2name(self):
        self.obj_class2name = {}
        for name, class_idx in self.obj_name2class.items():
            self.obj_class2name[class_idx] = name

    def get_obj_class(self, obj_name):
        return self.obj_name2class[obj_name]

    def __str__(self):
        x = {}
        for attr in self.save_attributes:
            x[attr] = getattr(self, attr)
        return str(x)

    @property
    def num_obj_class(self):
        return len(self.obj_class2name)

    def is_obj_token(self, obj):
        if isinstance(obj, str):
            return obj in self.obj_tokens
        try:
            obj_class = int(obj)
            return self.obj_class2name[obj_class] in self.obj_tokens
        except:
            return ValueError
        return ValueError

    def __len__(self):
        return self.num_obj_class

    def save(self, fname):
        x = {}
        for attr in self.save_attributes:
            x[attr] = getattr(self, attr)
        with open(fname, 'w') as f:
            json.dump(x, f, indent=4)

    def load(self, fname):
        with open(fname, 'r') as f:
            x = json.load(f)
        for attr in self.save_attributes:
            setattr(self, attr, x[attr])
            # for key in x[attr]:
        self.obj_name2color = { k: tuple(v) for k, v in self.obj_name2color.items() }
