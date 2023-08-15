# -*- coding: utf-8 -*-
import argparse
import os
import random
import time
import warnings

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from data_generator import MiniImagenet, ISIC, DermNet
from learner import Conv_Standard
from protonet import Protonet
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='MLTI')
parser.add_argument('--datasource', default='miniimagenet', type=str,
                    help='miniimagenet, isic, dermnet')
parser.add_argument('--num_classes', default=5, type=int,
                    help='number of classes used in classification (e.g. 5-way classification).')
# parser.add_argument('--test_epoch', default=-1, type=int, help='test epoch, only work when test start')
parser.add_argument('--test_epoch_start', default=500, type=int, help='test epoch, only work when test start')
parser.add_argument('--test_epoch_end', default=50500, type=int, help='test epoch, only work when test end')
## Training options
parser.add_argument('--metatrain_iterations', default=50000, type=int,
                    help='number of metatraining iterations.')  # 15k for omniglot, 50k for sinusoid
parser.add_argument('--meta_batch_size', default=4, type=int, help='number of tasks sampled per meta-update')  # 25
parser.add_argument('--meta_lr', default=0.001, type=float, help='the base learning rate of the generator')
parser.add_argument('--update_batch_size', default=1, type=int,  # 5
                    help='number of examples used for inner gradient update (K for K-shot learning).')
parser.add_argument('--update_batch_size_eval', default=15, type=int,
                    help='number of examples used for inner gradient test (K for K-shot learning).')

## Model options
parser.add_argument('--num_filters', default=32, type=int,
                    help='number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')

## Logging, saving, and testing options
parser.add_argument('--logdir', default=os.path.abspath('..') + '/out/', type=str,
                    help='directory for summaries and checkpoints.')
parser.add_argument('--datadir', default=os.path.abspath('..') + '/data/', type=str, help='directory for datasets.')
parser.add_argument('--resume', default=0, type=int, help='resume training if there is a model available')
parser.add_argument('--train', default=1, type=int, help='True to train, False to test.')
parser.add_argument('--mix', default=1, type=int, help='use mixup or not')
parser.add_argument('--trial', default=0, type=int, help='trail for each layer')
parser.add_argument('--ratio', default=0.2, type=float, help='the ratio of meta-training tasks')
parser.add_argument('--beta', default=0, type=float, help='the ratio of org loss')
parser.add_argument('--lam', default=0.001, type=float, help='the ratio of org loss')

args = parser.parse_args()

if args.datasource == 'isic':
    assert args.num_classes < 5

assert torch.cuda.is_available()
torch.backends.cudnn.benchmark = True

random.seed(1)
np.random.seed(2)

exp_string = 'ProtoNet_Cross' + '.data_' + str(args.datasource) + '.cls_' + str(args.num_classes) + '.mbs_' + str(
    args.meta_batch_size) + '.ubs_' + str(
    args.update_batch_size) + '.metalr' + str(args.meta_lr)

if args.num_filters != 64:
    exp_string += '.hidden' + str(args.num_filters)
if args.mix:
    exp_string += '.mix'
if args.trial > 0:
    exp_string += '.trial{}'.format(args.trial)
if args.ratio < 1.0:
    exp_string += '.ratio{}'.format(args.ratio)

if args.beta > 0:
    exp_string += '.beta{}'.format(args.beta)

print(exp_string)

writer = SummaryWriter(log_dir='./runs/' + exp_string)


def print_and_log(log_file, message):
    print(message)
    log_file.write(message + '\n')


def train(args, protonet, optimiser):
    Print_Iter = 100
    Save_Iter = 500
    print_loss, print_acc = 0.0, 0.0

    if args.datasource == 'miniimagenet':
        dataloader = MiniImagenet(args, 'train')
    elif args.datasource == 'isic':
        dataloader = ISIC(args, 'train')
    elif args.datasource == 'dermnet':
        dataloader = DermNet(args, 'train')

    if not os.path.exists(args.logdir + '/' + exp_string + '/'):
        os.makedirs(args.logdir + '/' + exp_string + '/')

    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(dataloader):


        if step > args.metatrain_iterations:
            break

        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).cuda(), y_spt.squeeze(0).cuda(), \
                                     x_qry.squeeze(0).cuda(), y_qry.squeeze(0).cuda()
        task_losses = []
        task_acc = []

        for meta_batch in range(args.meta_batch_size):

            if args.mix:
                mix_c = random.randint(0, args.meta_batch_size)
                if mix_c > 0:

                    second_id = (meta_batch + 1) % args.meta_batch_size

                    loss_val_mix, acc_val, cbn_kl_loss = protonet.forward_crossmix(x_spt[meta_batch], y_spt[meta_batch],
                                                                                   x_qry[meta_batch],
                                                                                   y_qry[meta_batch],
                                                                                   x_spt[second_id], y_spt[second_id],
                                                                                   x_qry[second_id],
                                                                                   y_qry[second_id])
                    loss_val_org, _ = protonet(x_spt[meta_batch], y_spt[meta_batch],
                                               x_qry[meta_batch],
                                               y_qry[meta_batch], )
                    loss_val = loss_val_mix + args.beta * loss_val_org + args.lam * cbn_kl_loss

                else:
                    loss_val_mix, acc_val, cbn_kl_loss = protonet.forward_crossmix(x_spt[meta_batch], y_spt[meta_batch],
                                                                                   x_qry[meta_batch],
                                                                                   y_qry[meta_batch],
                                                                                   x_spt[meta_batch], y_spt[meta_batch],
                                                                                   x_qry[meta_batch],
                                                                                   y_qry[meta_batch])
                    loss_val_org, _ = protonet(x_spt[meta_batch], y_spt[meta_batch],
                                               x_qry[meta_batch],
                                               y_qry[meta_batch], )
                    loss_val = loss_val_mix + args.beta * loss_val_org + args.lam * cbn_kl_loss

            else:
                loss_val, acc_val = protonet(x_spt[meta_batch], y_spt[meta_batch], x_qry[meta_batch], y_qry[meta_batch])

            task_losses.append(loss_val)
            task_acc.append(acc_val)



        meta_batch_loss = torch.stack(task_losses).mean()
        meta_batch_acc = torch.stack(task_acc).mean()

        optimiser.zero_grad()
        meta_batch_loss.backward()
        optimiser.step()

        if step != 0 and step % Print_Iter == 0:
            message = 'step:{}, loss:{}, acc:{}, {}'.format(step, print_loss, print_acc,
                                                            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            writer.add_scalar('Train Loss', print_loss, step)
            writer.add_scalar('Train Accuracy', print_acc, step)
            writer.close()

            log_file = open(args.logdir + '/' + exp_string + '/' + 'log.txt', "a")
            print_and_log(log_file, message)
            print_loss, print_acc = 0.0, 0.0
        else:
            print_loss += meta_batch_loss / Print_Iter
            print_acc += meta_batch_acc / Print_Iter

        if step != 0 and step % Save_Iter == 0:
            torch.save(protonet.learner.state_dict(),
                       '{0}/{2}/model{1}'.format(args.logdir, step, exp_string))


def test(args, protonet, test_step):
    protonet.eval()
    res_acc = []
    args.meta_batch_size = 1

    if args.datasource == 'miniimagenet':
        dataloader = MiniImagenet(args, 'test')
    elif args.datasource == 'isic':
        dataloader = ISIC(args, 'test')
    elif args.datasource == 'dermnet':
        dataloader = DermNet(args, 'test')

    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(dataloader):
        if step > 600:
            break
        if args.datasource in ['isic', 'dermnet']:
            x_spt, y_spt, x_qry, y_qry = x_spt.to("cuda"), y_spt.to("cuda"), \
                                         x_qry.to("cuda"), y_qry.to("cuda")
        else:
            x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to("cuda"), y_spt.squeeze(0).to("cuda"), \
                                         x_qry.squeeze(0).to("cuda"), y_qry.squeeze(0).to("cuda")

        with torch.no_grad():
            _, acc_val = protonet(x_spt, y_spt, x_qry, y_qry)
            res_acc.append(acc_val.item())

    res_acc = np.array(res_acc)

    message = 'test_epoch is {}, acc is {}, ci95 is {}'.format(test_step, np.mean(res_acc),
                                                               1.96 * np.std(res_acc) / np.sqrt(
                                                                   600 * args.meta_batch_size))

    log_file = open(args.logdir + '/' + exp_string + '/' + 'test_log.txt', "a")
    print_and_log(log_file, message)


def main():
    start = time
    learner = Conv_Standard(args=args, x_dim=3, hid_dim=args.num_filters, z_dim=args.num_filters).cuda()

    protonet = Protonet(args, learner)

    if args.resume == 1 and args.train == 1:
        model_file = '{0}/{2}/model{1}'.format(args.logdir, args.test_epoch, exp_string)
        print(model_file)
        learner.load_state_dict(torch.load(model_file))

    meta_optimiser = torch.optim.Adam(list(learner.parameters()),
                                      lr=args.meta_lr, weight_decay=args.weight_decay)

    if args.train == 1:
        train(args, protonet, meta_optimiser)
    else:
        for test_step in range(args.test_epoch_start, args.test_epoch_end, 500):
            model_file = '{0}/{2}/model{1}'.format(args.logdir, test_step, exp_string)
            protonet.learner.load_state_dict(torch.load(model_file), False)
            test(args, protonet, test_step)


if __name__ == '__main__':
    main()
