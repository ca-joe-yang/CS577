import os
from data.coco import CocoDataset
import argparse
import torch
import numpy as np
import random
import torch
from utils import set_random_seed
from models import Encoder, DecoderWithAttention

import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from trainer import train, validate

parser = argparse.ArgumentParser()
# parser.add_argument('--dataloader', default='op_vg_msdn', type=str)
parser.add_argument('--dataset', default='coco', type=str)
parser.add_argument('--dataroot', default='datasets/mscoco', type=str)
parser.add_argument('--num_workers', default=1, type=int)
parser.add_argument('--max_caplen', default=15, type=int)
parser.add_argument('--min_word_freq', default=5, type=int)
parser.add_argument('--min_object_size', default=0.02, type=float)
parser.add_argument('--min_objects', default=3, type=int)
parser.add_argument('--max_objects', default=8, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--fine_tune_encoder', action='store_true')
parser.add_argument('--encoder_lr', default=1e-4, type=float)
parser.add_argument('--decoder_lr', default=4e-4, type=float)
parser.add_argument('--emb_dim', default=512, type=int)
parser.add_argument('--attention_dim', default=512, type=int)
parser.add_argument('--decoder_dim', default=512, type=int)
parser.add_argument('--dropout', default=0.5, type=float)
parser.add_argument('--print_freq', default=1, type=int)
parser.add_argument('--alpha_c', default=1.0, type=float)

# fine_tune_encoder = False  # fine-tune encoder?

args = parser.parse_args()
device = torch.device("cuda")  # sets device for model and PyTorch tensors
# cudnn.benchmark = True

set_random_seed(0)

train_dataset = CocoDataset(
    args, split='train',
    image_dir=os.path.join(args.dataroot, "train2014"),
    instances_json_path=os.path.join(args.dataroot, "annotations/instances_train2014.json"),
    caption_json_path=os.path.join(args.dataroot, 'dataset_coco.json'),
    vocab_json_path=os.path.join(args.dataroot, "vocab.json"),
    wordmap_json_path=os.path.join(args.dataroot, "wordmap.json")
)
word_map = train_dataset.word_map

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    collate_fn=train_dataset.collate_fn,
)

# global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map

# # Read word map
# word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
# with open(word_map_file, 'r') as j:
#     word_map = json.load(j)
# print(word_map_file)

# Initialize / load checkpoint
decoder = DecoderWithAttention(attention_dim=args.attention_dim,
                                embed_dim=args.emb_dim,
                                decoder_dim=args.decoder_dim,
                                vocab_size=len(word_map),
                                dropout=args.dropout)
# decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
#                                         lr=args.decoder_lr)
encoder = Encoder()
# encoder.fine_tune(args.fine_tune_encoder)
# encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
#                                         lr=args.encoder_lr) if args.fine_tune_encoder else None

# # Move to GPU, if available
decoder = decoder.to(device)
encoder = encoder.to(device)
print('build')
# Loss function
criterion = nn.CrossEntropyLoss().to(device)

# Custom dataloaders
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

best_bleu4 = 0.
start_epoch = 0
epochs = 120  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0 
# Epochs
for epoch in range(start_epoch, epochs):
    print(epoch)
    # # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
    # if epochs_since_improvement == 20:
    #     break
    # if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
    #     adjust_learning_rate(decoder_optimizer, 0.8)
    #     if fine_tune_encoder:
    #         adjust_learning_rate(encoder_optimizer, 0.8)

    # One epoch's training
    print('train')
    train(
        train_loader=train_loader, 
        encoder=encoder, decoder=decoder,
        criterion=criterion,
        epoch=epoch,
        device=device
    )
    # train(train_loader=train_loader,
    #         encoder=encoder,
    #         decoder=decoder,
    #         criterion=criterion,
    #         encoder_optimizer=encoder_optimizer,
    #         decoder_optimizer=decoder_optimizer,
    #         epoch=epoch)

    # One epoch's validation
    recent_bleu4 = validate(val_loader=val_loader,
                            encoder=encoder,
                            decoder=decoder,
                            criterion=criterion)

    # Check if there was an improvement
    print(recent_bleu4)
    is_best = recent_bleu4 > best_bleu4
    best_bleu4 = max(recent_bleu4, best_bleu4)
    if not is_best:
        epochs_since_improvement += 1
        print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
    else:
        epochs_since_improvement = 0

    # Save checkpoint
    save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                    decoder_optimizer, recent_bleu4, is_best)

# for inputs in train_loader:
#     print(inputs)
#     # break