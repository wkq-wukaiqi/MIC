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
from dalib.modules.teacher import EMATeacher
from common.utils.data import ForeverDataIterator
from common.utils.metric import accuracy
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger
from common.utils.analysis import collect_feature, tsne, a_distance
from common.utils.sam import SAM

sys.path.append('.')
import utils

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    x=x.float()
    y=y.float()
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        (dist_mat * is_pos.float()).contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    temp = dist_mat * is_neg.float()
    temp[temp == 0] = 10e5
    dist_an, relative_n_inds = torch.min(
        (temp).contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an

class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, feat, labels, normalize_feature=False):
        if normalize_feature:
            feat = normalize(feat, axis=-1)
        if len(feat.size()) == 3:
            raise NotImplementedError
        else:
            dist_mat = euclidean_dist(feat, feat)
        dist_ap, dist_an = hard_example_mining(
            dist_mat, labels)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss



def main(args: argparse.Namespace):
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
    classifier_feature_dim = classifier.features_dim

    if args.randomized:
        domain_discri = DomainDiscriminator(
            args.randomized_dim, hidden_size=1024).to(device)
    else:
        domain_discri = DomainDiscriminator(
            classifier_feature_dim * num_classes, hidden_size=1024).to(device)

    # define optimizer and lr scheduler
    base_optimizer = torch.optim.SGD
    ad_optimizer = SGD(domain_discri.get_parameters(
    ), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    optimizer = SAM(classifier.get_parameters(), base_optimizer, rho=args.rho, adaptive=False,
                    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr *
                            (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    lr_scheduler_ad = LambdaLR(
        ad_optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # define loss function
    domain_adv = ConditionalDomainAdversarialLoss(
        domain_discri, entropy_conditioning=args.entropy,
        num_classes=num_classes, features_dim=classifier_feature_dim, randomized=args.randomized,
        randomized_dim=args.randomized_dim
    ).to(device)

    mcc_loss = MinimumClassConfusionLoss(temperature=args.temperature)

    triplet_loss=TripletLoss(margin=0.3)

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

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(
            logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(
            classifier.backbone, classifier.pool_layer, classifier.bottleneck).to(device)
        source_feature = collect_feature(
            train_source_loader, feature_extractor, device)
        target_feature = collect_feature(
            train_target_loader, feature_extractor, device)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.pdf')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(
            source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return

    if args.phase == 'test':
        acc1 = utils.validate(test_loader, classifier, args, device)
        print(acc1)
        return

    # start training
    # classifier = nn.DataParallel(classifier)
    # teacher = nn.DataParallel(teacher)
    best_acc1 = 0.
    first_step_scaler=GradScaler()
    second_step_scaler=GradScaler()
    teacher = None
    for epoch in range(args.epochs):
        print("lr_bbone:", lr_scheduler.get_last_lr()[0])
        print("lr_btlnck:", lr_scheduler.get_last_lr()[1])
        if args.log_results:
            wandb.log({"lr_bbone": lr_scheduler.get_last_lr()[0],
                       "lr_btlnck": lr_scheduler.get_last_lr()[1]})
        # train for one epoch

        if teacher is None and (epoch + 1) >= args.teacher_epoch:
            print(f'Epoch {epoch + 1} add teacher')
            classifier.train()
            teacher = EMATeacher(classifier, alpha=args.alpha, pseudo_label_weight=args.pseudo_label_weight).to(device)

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

        train(first_step_scaler, second_step_scaler, train_source_iter, train_target_iter, classifier, teacher,
              domain_adv, mcc_loss, triplet_loss, masking_t, masking_s, optimizer, ad_optimizer,
              lr_scheduler, lr_scheduler_ad, epoch, args)
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
    # classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    # acc1 = utils.validate(test_loader, classifier, args, device)
    # print("test_acc1 = {:3.1f}".format(acc1))
    # if args.log_results:
    #     wandb.log({'epoch': epoch, 'test_acc': acc1})

    if args.log_results:
        wandb.finish()

    logger.close()


def train(first_step_scaler, second_step_scaler, train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
          model: ImageClassifier, teacher: EMATeacher,
          domain_adv: ConditionalDomainAdversarialLoss, mcc, triplet_loss, masking_t, masking_s, optimizer, ad_optimizer,
          lr_scheduler: LambdaLR, lr_scheduler_ad, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    trans_losses = AverageMeter('Trans Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    domain_accs = AverageMeter('Domain Acc', ':3.1f')
    pseudo_label_accs = AverageMeter('Pseudo Label Acc', ':3.1f')
    log_list = [batch_time, data_time, losses, trans_losses, cls_accs, domain_accs, pseudo_label_accs]
    if args.triplet:
        triplet_losses = AverageMeter('Triplet Loss', ':3.2f')
        log_list.append(triplet_losses)
    progress = ProgressMeter(
        args.iters_per_epoch,
        log_list,
        prefix="Epoch: [{}]".format(epoch+1))

    # switch to train mode
    model.train()
    domain_adv.train()

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
        ad_optimizer.zero_grad()

        # generate pseudo-label
        # teacher.module.update_weights(model, epoch * args.iters_per_epoch + i)
        if teacher is not None:
            teacher.update_weights(model, epoch * args.iters_per_epoch + i)
            pseudo_label_t, pseudo_prob_t, ema_softmax  = teacher(x_t)
            pseudo_label_acc, = accuracy(ema_softmax, labels_t, topk=(1,))

        # compute output
        with autocast():
            x = torch.cat((x_s, x_t), dim=0)
            # f是feature，y是预测
            y, f = model(x)
            y_s, y_t = y.chunk(2, dim=0)
            f_s, f_t = f.chunk(2, dim=0)
            cls_loss = F.cross_entropy(y_s, labels_s)
            # SDAT论文里的MinimumClassConfusionLoss，直接用
            mcc_loss_value = mcc(y_t)
            # mask数据的预测
            # y_t_masked, _ = model(x_t_masked)
            # if teacher.module.pseudo_label_weight is not None:
            # 用mask的预测结果和伪标签计算CE
            # masking_loss_value就是所谓的 MIC loss
            if teacher is not None:
                # 5轮后才加入teacher
                y_t_masked, _ = model(x_t_masked)
                if teacher.pseudo_label_weight is not None:
                    ce = F.cross_entropy(y_t_masked, pseudo_label_t, reduction='none')
                    masking_loss_value = torch.mean(pseudo_prob_t * ce)
                else:
                    masking_loss_value = F.cross_entropy(y_t_masked, pseudo_label_t)
                # 损失由源域数据分类损失、MCC损失、目标域mask后的分类损失（MIC loss）组成
                # 相当于是源域和目标域一起训练
                loss = cls_loss + mcc_loss_value + masking_loss_value
            else:
                loss = cls_loss + mcc_loss_value
            if args.triplet:
                triplet_loss_value = triplet_loss(f_s, labels_s) + triplet_loss(f_t, pseudo_label_t)
                loss = loss + triplet_loss_value

        first_step_scaler.scale(loss).backward()
        first_step_scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)

        optimizer_state = first_step_scaler._per_optimizer_states[id(optimizer)]

        # Check if any gradients are inf/nan
        inf_grad_cnt = sum(v.item() for v in optimizer_state["found_inf_per_device"].values())

        if inf_grad_cnt == 0:
            # if valid graident, apply sam_first_step
            optimizer.first_step(zero_grad=True)
            sam_first_step_applied = True
        else:
            # if invalid graident, skip sam and revert to single optimization step
            optimizer.zero_grad()
            sam_first_step_applied = False
        first_step_scaler.update()

        # Calculate task loss and domain loss
        with autocast():
            y, f = model(x)
            y_s, y_t = y.chunk(2, dim=0)
            f_s, f_t = f.chunk(2, dim=0)

            cls_loss = F.cross_entropy(y_s, labels_s)
            # y_t_masked, _ = model(x_t_masked)
            if teacher is not None:
                y_t_masked, _ = model(x_t_masked)
                transfer_loss = domain_adv(y_s, f_s, y_t, f_t) + mcc(y_t) + \
                            F.cross_entropy(y_t_masked, pseudo_label_t)
            else:
                transfer_loss = domain_adv(y_s, f_s, y_t, f_t) + mcc(y_t)
            domain_acc = domain_adv.domain_discriminator_accuracy
            # 来自SDAT的步骤，最终的损失除了上面的源域数据分类损失、MCC损失、MIC loss
            # 还有一个ConditionalDomainAdversarialLoss，即CDAN，这也是SDAT的一个组成部分
            # 简单来说就是一个对抗训练，用于提取域不变特征
            if args.triplet:
                triplet_loss_value = triplet_loss(f_s, labels_s) + triplet_loss(f_t, pseudo_label_t)
                loss = triplet_loss_value + cls_loss + transfer_loss * args.trade_off
            else:
                loss = cls_loss + transfer_loss * args.trade_off            

        cls_acc = accuracy(y_s, labels_s)[0]
        if args.log_results:
            # masked_img = wandb.Image(x_t_masked, caption="Masked Image")
            wandb.log({
                'iteration': epoch*args.iters_per_epoch + i, 'loss': loss, 'cls_loss': cls_loss,
                'transfer_loss': transfer_loss, 'domain_acc': domain_acc, 'pseudo_weight_avg': torch.mean(pseudo_prob_t),
                # 'masked_img': masked_img,
           })

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc, x_s.size(0))
        domain_accs.update(domain_acc, x_s.size(0))
        trans_losses.update(transfer_loss.item(), x_s.size(0))
        if teacher is not None:
            pseudo_label_accs.update(pseudo_label_acc, x_s.size(0))
        if args.triplet:
            triplet_losses.update(triplet_loss_value.item(), x_s.size(0))

        second_step_scaler.scale(loss).backward()
        second_step_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)
        if sam_first_step_applied:
            optimizer.second_step()
        second_step_scaler.step(optimizer)
        second_step_scaler.step(ad_optimizer)
        second_step_scaler.update()

        lr_scheduler.step()
        lr_scheduler_ad.step()

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
    # 是否使用三元损失
    parser.add_argument('--triplet', action='store_true')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [args.gpu]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    main(args)
