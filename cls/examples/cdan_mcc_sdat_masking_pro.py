# Credits: https://github.com/thuml/Transfer-Learning-Library
import random
import time
import warnings
import sys
import argparse
import shutil
import os.path as osp
import os
import wandb
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.cuda.amp import autocast,GradScaler

sys.path.append('../')
from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.adaptation.cdan import ConditionalDomainAdversarialLoss, ImageClassifier
from dalib.adaptation.mcc import MinimumClassConfusionLoss
from dalib.modules.masking import Masking
from dalib.modules.teacher import EMATeacherPrototype
from common.utils.data import ForeverDataIterator
from common.utils.metric import accuracy
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger
from common.utils.analysis import collect_feature, tsne, a_distance
from common.utils.sam import SAM

sys.path.append('.')
import utils

def main(args: argparse.Namespace):
    assert args.resume_path is not None, 'baseline checkpoint required!'

    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.log_results:
        wandb.init(
            project="MIC",
            name=args.log_name)
        wandb.config.update(args)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True
    device = args.device

    # Data loading code
    train_transform = utils.get_train_transform(args.train_resizing, random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, 
                                                strong_aug=args.strong_aug_source, 
                                                color_jitter_s=args.mask_color_jitter_s,
                                                color_jitter_p=args.mask_color_jitter_p,
                                                blur=args.mask_blur,
                                                resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    train_target_transform = utils.get_train_target_transform(args.train_resizing, 
                                                random_horizontal_flip=not args.no_hflip,
                                                resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)
    print("train_source_transform: ", train_transform)
    print("train_target_transform: ", train_target_transform)
    print("val_transform: ", val_transform)

    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
        utils.get_dataset(args.data, args.root, args.source,
                          args.target, train_transform, val_transform, train_target_transform=train_target_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size*4, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size*4, shuffle=False, num_workers=args.workers)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    # print(backbone)
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                 pool_layer=pool_layer, finetune=not args.scratch).to(device)

    # 加载classifier的baseline模型
    baseline_dict = torch.load(args.resume_path)
    classifier.load_state_dict(baseline_dict)

    teacher = EMATeacherPrototype(classifier, 
                                  alpha=args.alpha, 
                                  pseudo_label_weight=args.pseudo_label_weight,
                                  threshold=args.pseudo_threshold).to(device)
    init_teacher(teacher, val_loader, device)

    # define optimizer and lr scheduler
    optimizer = torch.optim.SGD(classifier.get_parameters(), 
                                     lr=args.lr, 
                                     momentum=args.momentum, 
                                     weight_decay=args.weight_decay)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr *
                            (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    
    masking_t = Masking(
        block_size=args.mask_block_size,
        ratio=args.mask_ratio,
        color_jitter_s=args.mask_color_jitter_s,
        color_jitter_p=args.mask_color_jitter_p,
        blur=args.mask_blur,
        mean=args.norm_mean,
        std=args.norm_std)

    if args.mask_source:
        masking_s = Masking(
        block_size=args.mask_block_size,
        ratio=0.5,
        color_jitter_s=args.mask_color_jitter_s,
        color_jitter_p=args.mask_color_jitter_p,
        blur=args.mask_blur,
        mean=args.norm_mean,
        std=args.norm_std)
    else:
        masking_s = None

    # start training
    best_acc1 = 0.
    scaler=GradScaler()
    for epoch in range(args.epochs):
        print("lr_bbone:", lr_scheduler.get_last_lr()[0])
        print("lr_btlnck:", lr_scheduler.get_last_lr()[1])
        if args.log_results:
            wandb.log({"lr_bbone": lr_scheduler.get_last_lr()[0],
                       "lr_btlnck": lr_scheduler.get_last_lr()[1]})

        if args.dynamic_mratio:
            if epoch < 8:
                ratio = 0.7
            elif epoch < 12:
                ratio = 0.7
            elif epoch < 20:
                ratio = 0.5
            else:
                ratio = 0.3
            print(f'Epoch[{epoch + 1}] Update Masking Ratio:{ratio}')
            masking_t.update_ratio(ratio)

        # train for one epoch
        train(scaler, train_source_iter, train_target_iter, classifier, teacher,
              masking_t, masking_s, optimizer, lr_scheduler, epoch, args)
        
        # evaluate on validation set
        acc1 = utils.validate(val_loader, classifier, args, device)
        if args.log_results:
            wandb.log({'epoch': epoch, 'val_acc': acc1})

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(),
                   logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'),
                        logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = utils.validate(test_loader, classifier, args, device)
    print("test_acc1 = {:3.1f}".format(acc1))
    if args.log_results:
        wandb.log({'epoch': epoch, 'test_acc': acc1})

    if args.log_results:
        wandb.finish()

    logger.close()

def init_teacher(teacher, val_loader, device):
    """
    初始化teacher，计算原型
    """
    loop = tqdm(enumerate(val_loader), total=len(val_loader))
    loop.set_description(f'Initializing Teacher...')
    teacher.init_begin()
    pseudo_label_usages = AverageMeter('Pseudo Usage', ':3.3f')
    with torch.no_grad():
        for _, (images, _) in loop:
            images = images.to(device)
            pseudo_label_usage = teacher.init_prototypes(images)
            pseudo_label_usages.update(pseudo_label_usage / images.size(0), images.size(0))
    teacher.init_end()
    print(f'Pseudo Label Usage: {100*pseudo_label_usages.avg:.2f}%')


def train(scaler, train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
          model: ImageClassifier, teacher: EMATeacherPrototype,
          masking_t, masking_s, optimizer, lr_scheduler: LambdaLR,
          epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.3f')
    cls_losses = AverageMeter('Cls Loss', ':3.3f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    mask_losses = AverageMeter('Mask Loss', ':3.3f')
    kd_losses = AverageMeter('KD Loss', ':3.3f')
    pseudo_label_accs = AverageMeter('Pseudo Label Acc', ':3.1f')
    log_list = [batch_time, 
                data_time, 
                losses,
                cls_losses, 
                mask_losses, 
                kd_losses,
                pseudo_label_accs,
                cls_accs]

    progress = ProgressMeter(
        args.iters_per_epoch,
        log_list,
        prefix="Epoch: [{}]".format(epoch+1))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)
        # labels_t仅用于研究伪标签准确率
        x_t, labels_t = next(train_target_iter)

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        if args.mask_source:
            x_s = masking_s(x_s)

        # 生成target的mask
        x_t_masked = masking_t(x_t)
        labels_s = labels_s.to(device)

        labels_t = labels_t.to(device)

        # measure data loading time
        data_time.update(time.time() - end)
        optimizer.zero_grad()

        # generate pseudo-label
        teacher.update_weights(model, epoch * args.iters_per_epoch + i)
        pseudo_prob_t, pseudo_label_t, ema_softmax, features_teacher  = teacher(x_t)
        pseudo_label_acc, = accuracy(ema_softmax, labels_t, topk=(1,))

        # Calculate task loss and domain loss
        with autocast():
            x = torch.cat((x_s, x_t), dim=0)
            y, f = model(x)
            y_s, y_t = y.chunk(2, dim=0)
            f_s, f_t = f.chunk(2, dim=0)

            # 源域分类损失
            cls_loss = F.cross_entropy(y_s, labels_s)

            y_t_masked, f_t_masked = model(x_t_masked)

            # mask一致性损失
            if teacher.pseudo_label_weight is not None:
                ce = F.cross_entropy(y_t_masked, pseudo_label_t, reduction='none')
                masking_loss_value = torch.mean(pseudo_prob_t* ce)
            else:
                masking_loss_value = F.cross_entropy(y_t_masked, pseudo_label_t)

            # 一致性KL散度损失
            teacher_distance = torch.cdist(features_teacher, teacher.prototypes.detach(), p=2)
            # student_distance = torch.cdist(f_t, teacher.prototypes.detach(), p=2)
            student_distance_mask = torch.cdist(f_t_masked, teacher.prototypes.detach(), p=2)
            
            # kd_loss = F.kl_div(F.log_softmax(-student_distance, dim=1), F.softmax(-teacher_distance.detach(), dim=1)) + \
            #           F.kl_div(F.log_softmax(-student_distance_mask, dim=1), F.softmax(-teacher_distance.detach(), dim=1))
            
            kd_loss = F.kl_div(F.log_softmax(-student_distance_mask, dim=1), F.softmax(-teacher_distance.detach(), dim=1))

            # 总损失
            loss = cls_loss + 10*kd_loss + masking_loss_value

        cls_acc = accuracy(y_s, labels_s)[0]
        if args.log_results:
            # masked_img = wandb.Image(x_t_masked, caption="Masked Image")
            wandb.log({
                'iteration': epoch*args.iters_per_epoch + i, 'loss': loss, 'cls_loss': cls_loss,
                'kd_loss': kd_loss,
                'pseudo_weight_avg': torch.mean(pseudo_prob_t),
                # 'masked_img': masked_img,
           })

        losses.update(loss.item(), x_s.size(0))
        cls_losses.update(cls_loss.item(), x_s.size(0))
        cls_accs.update(cls_acc, x_s.size(0))
        pseudo_label_accs.update(pseudo_label_acc, x_s.size(0))
        mask_losses.update(masking_loss_value.item(), x_s.size(0))
        kd_losses.update(kd_loss.item(), x_s.size(0))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)
        scaler.step(optimizer)
        scaler.update()

        # 更新原型，按照论文代码，使用的是teacher的输出
        teacher.update_prototypes(features_teacher.detach(), pseudo_prob_t, pseudo_label_t)

        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CDAN+MCC with SDAT for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31', choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target', help='target domain(s)', nargs='+')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true',
                        help='whether train from scratch.')
    parser.add_argument('-r', '--randomized', action='store_true',
                        help='using randomized multi-linear-map (default: False)')
    parser.add_argument('-rd', '--randomized-dim', default=1024, type=int,
                        help='randomized dimension when using randomized multi-linear-map (default: 1024)')
    parser.add_argument('--entropy', default=False,
                        action='store_true', help='use entropy conditioning')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.005, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.001,
                        type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75,
                        type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9,
                        type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=40, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='cdan',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    parser.add_argument('--log_results', action='store_true',
                        help="To log results in wandb")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID")# 仅单卡
    parser.add_argument('--log_name', type=str,
                        default="log", help="log name for wandb")
    parser.add_argument('--rho', type=float, default=0.05, help="GPU ID")
    parser.add_argument('--temperature', default=2.0,
                        type=float, help='parameter temperature scaling')
    # masked image consistency
    parser.add_argument('--alpha', default=0.999, type=float)
    parser.add_argument('--pseudo_label_weight', default=None)
    parser.add_argument('--mask_block_size', default=32, type=int)
    parser.add_argument('--mask_ratio', default=0.5, type=float)
    parser.add_argument('--mask_color_jitter_s', default=0, type=float)
    parser.add_argument('--mask_color_jitter_p', default=0, type=float)
    parser.add_argument('--mask_blur', default=False, type=bool)

    # 开始加入teacher监督的轮次
    parser.add_argument('--teacher_epoch', default=1, type=int)
    # 源数据是否使用强增强
    parser.add_argument('--strong_aug_source', action='store_true')
    # 是否使用变化的mask_ratio
    parser.add_argument('--dynamic_mratio', action='store_true')
    # 是否给源域数据加上mask（mask自带strong aug）
    parser.add_argument('--mask_source', action='store_true')
    # baseline预训练模型地址
    parser.add_argument('--resume_path', default=None, type=str)
    # 伪标签置信度阈值
    parser.add_argument('--pseudo_threshold', default=0.9, type=float)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [args.gpu]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    main(args)