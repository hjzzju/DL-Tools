from dataset_pt import BaseDataset
from dataloader_pt import BaseDataLoader
from loss_pt import BaseLoss
from misc_pt import *
def get_optim(lr):
    # Lower the learning rate on the VGG fully connected layers by 1/10th. It's a hack, but it helps
    # stabilize the models.
    fc_params = [p for n,p in detector.named_parameters() if n.startswith('roi_fmap') and p.requires_grad]
    non_fc_params = [p for n,p in detector.named_parameters() if not n.startswith('roi_fmap') and p.requires_grad]
    params = [{'params': fc_params, 'lr': lr / 10.0}, {'params': non_fc_params}]
    # params = [p for n,p in detector.named_parameters() if p.requires_grad]

    if conf.adam:
        optimizer = optim.Adam(params, weight_decay=conf.adamwd, lr=lr, eps=1e-3)
    else:
        optimizer = optim.SGD(params, weight_decay=conf.l2, lr=lr, momentum=0.9)

    # scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1,
    #                               verbose=True, threshold=0.0001, threshold_mode='abs', cooldown=1)
    return optimizer #, scheduler

optimize = get_optim(lr*num_gpu*batch)

conf = config()
train, val, test = BaseDataset.splits(conf)

train_loader, val_loader, test_loader = BaseDataLoader.splits(train, val, test, conf)

net = Net()

#freeze the param
# for n, param in detector.detector.named_parameters():
#     param.requires_grad = False

# load the pretrained model
# ckpt = torch.load(conf.ckpt)

def train_epoch(epoch_num):
    tr = []
    for b, batch in enumerate(train_loader):
        tr.append(train_batch(batch))

    return tr

def train_batch(batch_data):
    result = net(batch_data)
    losses = {}
    criterion = BaseLoss()
    losses['...'] = criterion(result, labels)

    optimize.zero_grad()
    loss.backward()
    clip_grad_norm(
        [(n, p) for n, p in detector.named_parameters() if p.grad is not None],
        max_norm=conf.clip, verbose=verbose, clip=True)
    losses['total'] = loss
    optimize.step()

