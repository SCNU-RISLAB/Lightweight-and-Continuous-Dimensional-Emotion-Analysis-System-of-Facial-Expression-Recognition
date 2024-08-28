import os
import datetime
import time
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from MobileNetV3_pt import MobileNetV3_Large,ECA_MobileNetV3_Large,output_ECA_MobileNetV3_Large,output_FPN_MobileNetV3_Large,output_ECA_FPN_MobileNetV3_Large, ECA_MobileNetV3_Large_Plus, output_ECA_MobileNetV3_Large_Plus
import argparse
from utils import *

now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H-%M]-")
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def seed_torch(seed=1024):
    # Save random seed
    with open('seed.txt', 'w') as f:
        f.write(str(seed))
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()

parser = argparse.ArgumentParser(description='PyTorch Rafdb Training')
parser.add_argument('--datasets', type=str, default='Raf-DB')
parser.add_argument('--pretrain', type=bool, default=True)
parser.add_argument('--best_checkpoint_path', type=str, default='./your_path/'+time_str+'model_best.pth')
parser.add_argument('--model', type=str, default='FPN_output_MobileNetV3_Large', help='CNN architecture')
parser.add_argument('--epoch', default=120, type=int, help='number of epochs in training')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--imagesize', type=int, default=224, help='image size (default: 224)')
parser.add_argument('--crop_size', type=int, default=224, help='image crop size (default: 224)')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--num_classes', type=int, default=7, help='number of expressions (classes)')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

opt = parser.parse_args()

learning_rate_decay_start = 5
learning_rate_decay_every = 5  # Decay interval
learning_rate_decay_rate = 0.9  # Decay rate

txt_name = './cxm_out/' + time_str + 'log.txt'
with open(txt_name, 'a') as f:
    f.write('dataset:' + opt.datasets + '\n')
    f.write('Pretrain:' + str(opt.pretrain) + '\n')
    f.write('batch size: ' + str(opt.batch_size) + '\n')
    f.write('image size: ' + str(opt.imagesize) + '\n')
    f.write('crop size: ' + str(opt.crop_size) + '\n')
    f.write('model: ' + str(opt.model) + '\n')
    f.write('class: ' + str(opt.num_classes) + '\n')
    f.write('epoch: ' + str(opt.epoch) + '\n')
    f.write('learning_rate_decay_start = 30' + '\n')
    f.write('learning_rate_decay_every' + '\n')
    f.write('************************************************' + '\n')

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def adjust_learning_rate(optimizer, epoch):
    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        exp = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** exp
        current_lr = opt.lr * decay_factor
        set_lr(optimizer, current_lr)  # Set the decayed rate
    else:
        current_lr = opt.lr

    with open(txt_name, 'a') as f:
        f.write('Current learning rate: ' + str(current_lr) + '\n')

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)


def main(opt):
    best_acc = 0
    best_epoch = 0
    now = datetime.now()
    print('Training time:', now.strftime("%m-%d %H:%M"))

    print('==> Building model..')
    model_classes = {
        'MobileNetV3_Large': MobileNetV3_Large,
        'ECA_MobileNetV3_Large': ECA_MobileNetV3_Large,
        'ECA_MobileNetV3_Large_Plus': ECA_MobileNetV3_Large_Plus,
        'output_ECA_MobileNetV3_Large': output_ECA_MobileNetV3_Large,
        'output_ECA_MobileNetV3_Large_Plus': output_ECA_MobileNetV3_Large_Plus,
        'output_FPN_MobileNetV3_Large': output_FPN_MobileNetV3_Large,
        'output_ECA_FPN_MobileNetV3_Large': output_ECA_FPN_MobileNetV3_Large


    }

    model = model_classes[opt.model](num_classes=7)
    print(f'Model is {opt.model}')

    model = torch.nn.DataParallel(model).cuda()

    if opt.pretrain:
        checkpoint = torch.load(opt.pretrain_path)
        model.load_state_dict(checkpoint['state_dict'])

    criterion = nn.CrossEntropyLoss().cuda()
    lsce_criterion = LabelSmoothingCrossEntropy(smoothing=0.2)
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)

    train_transform = transforms.Compose([
        transforms.Resize((opt.imagesize, opt.imagesize)),
        transforms.RandomCrop(opt.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((opt.imagesize, opt.imagesize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print('==> Preparing data..')
    datasets = {
        "Raf-DB": ('/path/to/Raf/train', '/path/to/Raf/test'),
        "FER2013": ('/path/to/fer2013_plus/Training', '/path/to/fer2013_plus/PrivateTest')
    }

    train_data_path, test_data_path = datasets[opt.datasets]

    train_datasets = ImageFolder(train_data_path, transform=train_transform)
    train_dataloader = DataLoader(train_datasets, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers,
                                  pin_memory=True)

    test_datasets = ImageFolder(test_data_path, transform=test_transform)
    test_dataloader = DataLoader(test_datasets, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers,
                                 pin_memory=True)

    print(f'Length of {opt.datasets} train Database: {len(train_datasets)}')
    print(f'Length of {opt.datasets} test Database: {len(test_datasets)}')

    for epoch in range(opt.start_epoch, opt.epoch):
        start_time = time.time()
        lr = optimizer.param_groups[0]['lr']
        print('Current learning rate:', lr)
        adjust_learning_rate(optimizer, epoch)

        train_acc, train_loss = train(train_dataloader, model, criterion, lsce_criterion, optimizer, epoch, opt)
        print(f'Train {epoch + 1} epoch | train_Loss: {train_loss:.6f} | train_Acc: {train_acc:.6f}')

        val_acc, val_loss = test(test_dataloader, model, criterion, opt)
        print(f'Test {epoch + 1} epoch | test_Loss: {val_loss:.6f} | test_Acc: {val_acc:.6f}')

        is_best = val_acc > best_acc
        if is_best:
            best_epoch = epoch + 1
        best_acc = max(val_acc, best_acc)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, opt)

        print('Current best accuracy:', best_acc.item())
        print('Current best epoch:', best_epoch)

        end_time = time.time()
        epoch_time = end_time - start_time
        print(f"An Epoch Time: {epoch_time:.6f}")
        print('---------------------------------------------')

def train(train_loader, model, criterion, optimizer):
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    for i, (images, target) in enumerate(train_loader):
        images = images.cuda()
        target = target.cuda()
        images, target = Variable(images), Variable(target)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, _ = accuracy(output, target, topk=(1,5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
            acc, _ = accuracy(output, target, topk=(1,5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc[0], images.size(0))

    return top1.avg, losses.avg

def save_checkpoint(state, is_best, args):
    if is_best:
        torch.save(state, args.best_checkpoint_path)
        print('best model save')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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

class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 30
        self.epoch_losses[idx, 1] = val_loss * 30
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):
        # Visualization of accuracy/loss curve for training/validation

        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1800, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 5
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

if __name__ == '__main__':
    main()
    print('RAF-DB train finish')















