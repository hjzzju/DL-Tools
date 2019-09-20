import json

import matplotlib
from scipy.signal import medfilt
from tqdm import tqdm

import lrschedule

matplotlib.use("Agg")
import os
import time
import argparse
import math
from numpy import finfo

import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from model import Tacotron2, TacotronAsr
from data_utils import TextMelLoader, TextMelCollate, PartialyRandomizedSimilarTimeLengthSampler
from loss_function import Tacotron2Loss, TacotronAsrLoss
from logger import Tacotron2Logger
from hparams import create_hparams
import numpy as np
import glob

from utils import get_diagonal_mask


def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)

    print("Done initializing distributed")


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader('train', hparams)
    valset = TextMelLoader('val', hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step, hparams)

    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset)
    else:
        l = trainset.get_lengths()
        train_sampler = PartialyRandomizedSimilarTimeLengthSampler(
            l, batch_size=hparams.batch_size * (sum(l) / len(l)))

    train_loader = DataLoader(trainset, num_workers=24,
                              shuffle=False,
                              sampler=train_sampler,
                              batch_size=1,
                              pin_memory=False,
                              drop_last=False,
                              collate_fn=collate_fn)
    return train_loader, valset, collate_fn


def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger


def load_model(hparams):
    if hparams.reverse:
        model = TacotronAsr(hparams).cuda()
    else:
        model = Tacotron2(hparams).cuda()
    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)
    return model


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    # if len(ignore_layers) > 0:
    #     model_dict = {k: v for k, v in model_dict.items()
    #                   if k not in ignore_layers}
    #     dummy_dict = model.state_dict()
    #     dummy_dict.update(model_dict)
    #     model_dict = dummy_dict
    model.load_state_dict(model_dict, strict=False)
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    # checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'], strict=False)
    try:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    except Exception as e:
        print("optimizer params load failed", e)
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath, output_directory):
    ckpt_paths = glob.glob(os.path.join(output_directory, 'checkpoint_*'))
    if len(ckpt_paths) > 0:
        for ck in sorted(ckpt_paths, key=lambda x: int(x.split("_")[-1]))[:-10]:
            os.remove(ck)
            print("remove ckpt:", ck)

    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def validate(model, criterion, valset, iteration, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank, hparams, output_directory):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=4,
                                shuffle=False,
                                batch_size=batch_size * 10,
                                pin_memory=False, collate_fn=collate_fn, drop_last=False)

        val_loss = 0.0
        gta_val_loss = 0.0
        for i, batch in enumerate(val_loader):
            assert i == 0  # only one test batch supported
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            attn_mask = x[10]
            losses = criterion(y_pred, y, x)
            reduced_gta_val_loss = sum(losses.values())
            if distributed_run:
                reduced_gta_val_loss = reduce_tensor(reduced_gta_val_loss.data, n_gpus).item()
            else:
                reduced_gta_val_loss = reduced_gta_val_loss.item()
            gta_val_loss += reduced_gta_val_loss

            y_pred = y_pred[:5]
            input_lengths = x[1]
            output_lengths = x[4]
            if hparams.do_infer:
                y_infer = model.inference(x)
                y_pred[:3] = y_infer[:3]
                if len(y_infer) > 4:
                    y_pred.append(y_infer[4])
            else:
                losses = criterion(y_pred, y, x)
                reduced_val_loss = sum(losses.values())
                if distributed_run:
                    reduced_val_loss = reduce_tensor(reduced_val_loss.data, n_gpus).item()
                else:
                    reduced_val_loss = reduced_val_loss.item()
                val_loss += reduced_val_loss

            if logger is not None:
                logger.log_validation(x[7], attn_mask, model, y, y_pred, input_lengths, output_lengths, iteration,
                                      hparams.reverse, hparams)
        logger.add_scalar("validate.val_loss", val_loss, iteration)
        logger.add_scalar("validate.gta_val_loss", gta_val_loss, iteration)

    model.train()

    if rank == 0:
        print("Validation loss {}: {:9f}  {}".format(iteration, val_loss, "".join(
            ["[{}]:{:.4f}".format(k, v.item()) for k, v in losses.items()])))


def train(output_directory, log_directory, checkpoint_path, warm_start, n_gpus,
          rank, group_name, hparams):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """
    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)

    if hparams.fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O2')

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    if hparams.reverse:
        criterion = TacotronAsrLoss(hparams)
    else:
        criterion = Tacotron2Loss(hparams)

    logger = prepare_directories_and_logger(
        output_directory, log_directory, rank)

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0

    def load_ckpt(checkpoint_path, model, optimizer):
        model, optimizer, _learning_rate, iteration = load_checkpoint(
            checkpoint_path, model, optimizer)
        if hparams.use_saved_learning_rate:
            learning_rate = _learning_rate
        else:
            learning_rate = hparams.learning_rate
        iteration += 1  # next iteration is iteration + 1
        epoch_offset = max(0, int(iteration / len(train_loader)))
        return model, optimizer, learning_rate, iteration, epoch_offset

    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(
                checkpoint_path, model, hparams.ignore_layers)
        else:
            model, optimizer, learning_rate, iteration, epoch_offset = load_ckpt(checkpoint_path, model, optimizer)
    else:
        ckpt_paths = glob.glob(os.path.join(output_directory, 'checkpoint_*'))
        if len(ckpt_paths) > 0:
            last_ckpt_path = sorted(ckpt_paths, key=lambda x: int(x.split("_")[-1]))[-1]
            model, optimizer, learning_rate, iteration, epoch_offset = load_ckpt(last_ckpt_path, model, optimizer)

    # print(">>>>", model.wavenet.first_conv.weight.data[0])
    model.train()
    if hparams.save_mels:
        model.eval()
    print(model)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    # if hparams.full_song:
    #     splits_fn = os.path.join(output_directory, 'splits_info.json')
    #     if not os.path.exists(splits_fn):
    #         splits_info = {}
    #         for idx in np.load(hparams.ds_name + '.npz', allow_pickle=True).keys():
    #             splits_info[idx] = [[0, 0]]
    #         with open(splits_fn, 'w') as f:
    #             json.dump(splits_info, f)

    is_overflow = False
    # ================ MAIN TRAINNIG LOOP! ===================
    all_alignments = {}
    all_mels = {}
    all_linears = {}
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))
        t = tqdm(enumerate(train_loader))
        all_reduced_loss = 0
        for i, batch in t:
            current_lr = hparams.learning_rate
            if hparams.lr_schedule is not None:
                lr_schedule_f = getattr(lrschedule, hparams.lr_schedule)
                current_lr = lr_schedule_f(
                    hparams.learning_rate, iteration, **hparams.lr_schedule_kwargs)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

            if hparams.test_mode or (not is_overflow and (iteration % hparams.iters_per_checkpoint == 0)):
                if hparams.do_validation:
                    validate(model, criterion, valset, iteration,
                         hparams.batch_size, n_gpus, collate_fn, logger,
                         hparams.distributed_run, rank, hparams, output_directory)
                if hparams.test_mode:
                    exit()
                if rank == 0:
                    checkpoint_path = os.path.join(
                        output_directory, "checkpoint_{}".format(iteration))
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    checkpoint_path, output_directory)

            start = time.perf_counter()
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            model.zero_grad()
            x, y = model.parse_batch(batch)
            y_pred = model(x, iteration)
            losses = criterion(y_pred, y, x)
            loss = sum(losses.values())
            if hparams.distributed_run:
                reduced_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_loss = loss.item()
            if not hparams.save_mels:
                if hparams.fp16_run:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

            if hparams.fp16_run:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), hparams.grad_clip_thresh)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), hparams.grad_clip_thresh)
            is_overflow = math.isnan(grad_norm)

            if not is_overflow:
                optimizer.step()
            else:
                optimizer.zero_grad()
                print(loss, "grad overflow!!")

            input_lengths, output_lengths, uttids, mel_outputs, linear_outputs, alignments = \
                x[1], x[4], x[7], y_pred[1], y_pred[2], y_pred[3]
            if rank == 0:
                # if not is_overflow and rank == 0:
                duration = time.perf_counter() - start
                all_reduced_loss += reduced_loss

                t.set_description("iter:{},loss:{:.6f},GN:{:.6f},{:.2f}s/it,lr:{:.6f},"
                                  "details:{},shape:{}".format(
                    iteration, all_reduced_loss / (i + 1), grad_norm, duration, current_lr,
                    "".join(["[{}]:{:.4f}".format(k, v.item()) for k, v in losses.items()]),
                    list(mel_outputs.data.shape)))

                logger.log_training(
                    reduced_loss, grad_norm, learning_rate, duration, iteration)
            iteration += 1

            # save alignments
            input_lengths = input_lengths.data.cpu().numpy()
            output_lengths = output_lengths.data.cpu().numpy()
            uttids = uttids.data.cpu().numpy()
            if hparams.save_attn:
                alignments = alignments.data.cpu().numpy()
                for uttid, alignment, input_length, output_length \
                        in zip(uttids, alignments, input_lengths, output_lengths):
                    if hparams.reverse:
                        all_alignments[str(uttid)] = alignment[:input_length, :output_length]
                    else:
                        all_alignments[str(uttid)] = alignment[:output_length, :input_length]

            if hparams.save_mels:
                mel_outputs = mel_outputs.data.cpu().numpy()
                linear_outputs = linear_outputs.data.cpu().numpy()
                for uttid, mel_output, linear_output, input_length, output_length \
                        in zip(uttids, mel_outputs, linear_outputs, input_lengths, output_lengths):
                    all_mels[str(uttid)] = mel_output[:, :output_length]
                    all_linears[str(uttid)] = linear_output[:, :output_length]

        if hparams.save_attn:
            np.savez(os.path.join(output_directory, "all_alignments"), **all_alignments)
            exit()
        if hparams.save_mels:
            np.savez(os.path.join(output_directory, "all_mels"), **all_mels)
            np.savez(os.path.join(output_directory, "all_linears"), **all_linears)
            exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs', default='logdir')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')
    parser.add_argument('--save_attn', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--speech', action='store_true')

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)
    hparams.output_directory = args.output_directory
    if args.test:
        hparams.iters_per_checkpoint = 2
        hparams.test_mode = True

    if args.speech or hparams.speech:
        hparams.speech = True
        hparams.use_pitch = False
        hparams.use_ali = False
        hparams.multispeaker = True
        hparams.always_teacher_forcing = True
        hparams.attn_at_first = False
        hparams.linear_use_eo = False
        hparams.do_infer = False
        if 'libris' in hparams.ds_name:
            hparams.hop_size = 256
            hparams.win_size = 1024
            hparams.n_fft = 1024
            hparams.num_freq = 513
            hparams.audio_sample_rate = 16000

    if args.save_attn:
        hparams.save_attn = args.save_attn

    if hparams.save_mels or hparams.save_attn:
        hparams.leave_val = 0
        hparams.do_validation = False

    if hparams.reverse:
        hparams.batch_size = 100
        hparams.use_ali = False
        hparams.use_pitch = False
        hparams.use_postnet = False
        hparams.use_linear = False
        if hparams.add_sil == 0:
            hparams.add_sil = 2
        hparams.encoder_res = True
        hparams.always_teacher_forcing = True
        hparams.leave_val = 0
        if hparams.use_delta_deltas:
            hparams.n_mel_channels *= 3
            hparams.audio_num_mel_bins *= 3

    if hparams.pitch_before:
        hparams.pitch_embed_dim = hparams.symbols_embedding_dim
    if hparams.use_wavenet:
        hparams.always_teacher_forcing = True
        hparams.wavenet_setting = True
        hparams.use_linear = False
        hparams.use_postnet = False
        hparams.batch_size = 16

    if hparams.wavenet_setting:
        hparams.hop_size = 256
        hparams.win_size = 1024
        hparams.n_fft = 1024
        hparams.fmin = 125
        hparams.fmax = 7600
        hparams.num_freq = 513
        hparams.preemphasize = False
        hparams.ds_name = "sb_singing_sent_wav_linear_ali_wavenet"
    if hparams.linear_directly:
        hparams.audio_num_mel_bins = hparams.n_mel_channels = hparams.num_freq
        hparams.use_postnet = False
        hparams.use_linear = False

    if args.hparams:
        hparams.parse(args.hparams)
    print(hparams.to_json())

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)
    print("Full song:", hparams.full_song)

    train(args.output_directory, args.log_directory, args.checkpoint_path,
          args.warm_start, args.n_gpus, args.rank, args.group_name, hparams)
