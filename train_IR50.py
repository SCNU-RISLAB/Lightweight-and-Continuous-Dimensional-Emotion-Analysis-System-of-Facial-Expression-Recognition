import random
import torch.utils.data as data
from torchvision import transforms
import os
import argparse
import collections
from data_preprocessing.dataset_raf import RafDataSet
import datetime
from time import time
from utils import *
from data_preprocessing.sam import SAM
from models.ir50 import Backbone


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='RAF_DB', help='dataset')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='Pytorch checkpoint file path')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--val_batch_size', type=int, default=64, help='Batch size for validation.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.00003, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=2, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs.')
    parser.add_argument('--gpu', type=str, default='0,1', help='assign multi-gpus by comma concat')
    return parser.parse_args()


def setup_seed(seed=1024):
    os.environ['PYTHONHASHSEED'] = str(seed)  # Set Python hash seed, disable hash randomization
    torch.manual_seed(seed)  # Set random seed for CPU
    torch.cuda.manual_seed(seed)  # Set random seed for current GPU
    torch.cuda.manual_seed_all(seed)  # Set random seed for all GPUs
    np.random.seed(seed)  # Set random seed for NumPy
    random.seed(seed)  # Set random seed for Python
    torch.backends.cudnn.deterministic = True  # Use deterministic convolution algorithms
    torch.backends.cudnn.benchmark = False  # Disable cuDNN's automatic tuner


def load_pretrained_weights(model, checkpoint):  # Load pretrained weights
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint  # Handle case where state_dict is not wrapped

    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    print('load_weight', len(matched_layers))
    return model


def run_training():
    args = parse_args()
    setup_seed()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print("Work on GPU: ", os.environ['CUDA_VISIBLE_DEVICES'])

    # Record experiments and results
    now_time = datetime.datetime.now()
    time_str = now_time.strftime("[%m-%d]-[%H-%M]-")
    txt_name = './log/' + time_str + args.dataset + '-' + 'log.txt'
    with open(txt_name, 'a+') as f:
        f.write('dataset: ' + str(args.dataset) + '\n')
        f.write('batch size: ' + str(args.batch_size) + '\n')
        f.write('epoch: ' + str(args.epochs) + '\n')
        f.write('lr: ' + str(args.lr) + '\n')
        f.write('optimizer: ' + str(args.optimizer) + '\n')
        f.write('**************Train Start!!!!**********************' + '\n')

    # Data processing
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.1)), ])
    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Create model
    if args.dataset == "RAF_DB":
        data_path = '/your_path/'
        train_dataset = RafDataSet(data_path, train=True, transform=data_transforms, basic_aug=True)
        val_dataset = RafDataSet(data_path, train=False, transform=data_transforms_val)
        model = Backbone(50, 0.0, 'ir')
    else:
        return print('dataset name is not correct')
    # Import weights
    ir_checkpoint = torch.load('./your_path.pth', map_location=lambda storage, loc: storage)
    model = load_pretrained_weights(model, ir_checkpoint)
    model = torch.nn.DataParallel(model).cuda()

    # Data loading
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.val_batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True)

    if args.checkpoint:
        print("Loading pretrained weights...", args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        checkpoint = checkpoint["model_state_dict"]
        model = load_pretrained_weights(model, checkpoint)

    # Optimizer, learning rate decay, loss function
    params = model.parameters()
    if args.optimizer == 'adamw':
        base_optimizer = torch.optim.AdamW
    elif args.optimizer == 'adam':
        base_optimizer = torch.optim.Adam
    elif args.optimizer == 'sgd':
        base_optimizer = torch.optim.SGD
    else:
        raise ValueError("Optimizer not supported.")
    optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, rho=0.05, adaptive=False, )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    CE_criterion = torch.nn.CrossEntropyLoss()
    lsce_criterion = LabelSmoothingCrossEntropy(smoothing=0.2)

    # Parameter collection
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Total Parameters: %.3fM' % parameters)

    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        start_time = time()
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        print('Current learning rate:', round(lr, 6))

        # Train one epoch
        train_acc, train_loss = train(train_loader, model, CE_criterion, lsce_criterion, optimizer, epoch, args)
        print('Train {} epoch | train_Loss: {:.6f} | train_Acc: {:.6f}'.format(epoch, train_loss, train_acc))
        elapsed = (time() - start_time) / 60
        print('Train time: %.2f' % elapsed)
        scheduler.step()  # Decay learning rate

        # Test one epoch
        val_acc, val_loss = test(val_loader, model, CE_criterion, args)
        print('test {} epoch | test_Loss: {:.6f} | test_Acc: {:.6f}'.format(epoch, val_loss, val_acc))
        if val_acc > best_acc:
            best_acc = val_acc
            print("best_acc:", best_acc.item())

        epoch_time = (time() - start_time) / 60
        print('epoch_time:%.2f' % epoch_time)
        print('---------------------------------------------')
        with open(txt_name, 'a') as f:
            f.write(' * epochs:' + str(epoch) + '\n')
            f.write(' * train_Accuracy: {top1.avg:.6f}'.format(train_acc) + '\n')
            f.write(' * train_loss:' + str(train_loss) + '\n')
            f.write(' * lr:' + str(lr) + '\n')
            f.write(' * Val_Accuracy:' + str(val_acc) + '\n')
            f.write(' * Val_loss:' + str(val_loss) + '\n')
            f.write('----------Epoch Time: ' + str(epoch_time) + '\n')

    print("The final best_acc:" + str(best_acc))


def train(train_loader, model, criterion, lsce_criterion, optimizer, epoch, args):
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    for i, (images, target) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        images, target = Variable(images), Variable(target)

        # compute output
        output = model(images)
        CE_loss = criterion(output, target)
        lsce_loss = lsce_criterion(output, target)
        loss = lsce_loss + CE_loss
        loss.backward()
        optimizer.first_step(zero_grad=True)

        # second forward-backward pass
        outputs = model(images)
        CE_loss = criterion(outputs, target)
        lsce_loss = lsce_criterion(outputs, target)

        loss = lsce_loss + CE_loss
        loss.backward()  # make sure to do a full forward pass
        optimizer.second_step(zero_grad=True)

        # measure accuracy and record loss
        acc1, _ = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

    return top1.avg, losses.avg


def test(val_loader, model, criterion, args):
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()
            images, target = Variable(images), Variable(target)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc, _ = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc[0], images.size(0))

    return top1.avg, losses.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self) -> object:
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    run_training()
    now = datetime.datetime.now()
    print(now)
