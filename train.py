'''tain'''
import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.nn.functional as F
from random import randint

from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from maskgct_utils import *
import safetensors

from module.data_utils import (
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
    DistributedBucketSampler,
)

cfg_path = "./config/maskgct.json"
device = "cuda:0"

semantic_code_ckpt = r'./MaskGCT_model/semantic_codec/model.safetensors'
codec_encoder_ckpt = r'./MaskGCT_model/acoustic_codec/model.safetensors'
codec_decoder_ckpt = r'./MaskGCT_model/acoustic_codec/model_1.safetensors'
t2s_model_ckpt = r'./MaskGCT_model/t2s_model/model.safetensors'
s2a_1layer_ckpt = r'./MaskGCT_model/s2a_model/s2a_model_1layer/model.safetensors'
s2a_full_ckpt = r'./MaskGCT_model/s2a_model/s2a_model_full/model.safetensors'

def train_VC(rank, n_gpus, hps):
    global global_step
    global batch_size
    global epochs
    global save_interval
    global print_interval
    global keep_training_epochs

    batch_size = hps.train.batch_size
    epochs = hps.train.epochs
    save_interval = hps.train.save_interval
    print_interval = hps.train.log_interval
    keep_training_epochs = hps.train.keep_training_epochs
    global_step = hps.train.keep_training_steps
    save_model_path = hps.train.save_model_path
    symbols = hps['symbols']

    keep_T2S_path = f'{save_model_path}/T2S_{keep_training_epochs}_{global_step}.safetensors'

    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    
    device = f'cuda:{rank}'

    cfg = load_config(cfg_path)
    semantic_model, semantic_mean, semantic_std = build_semantic_model(device)
    semantic_codec = build_semantic_codec(cfg.model.semantic_codec, device)
    t2s_model = build_t2s_model(cfg.model.t2s_model, device)

    # load semantic codec
    safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)

    if keep_training_epochs > 0:
        safetensors.torch.load_model(t2s_model, keep_T2S_path)
    if keep_training_epochs == -1:
        safetensors.torch.load_model(t2s_model, t2s_model_ckpt)
        keep_training_epochs = 0

    maskgct_inference_pipeline = MaskGCT_Inference_Pipeline(
        semantic_model,
        semantic_codec,
        None,
        None,
        t2s_model,
        None,
        None,
        semantic_mean,
        semantic_std,
        device,
    )

    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data, symbols)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        batch_size,
        [
            32, 300, 400, 500, 600, 700, 800, 900, 1000,
            1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900,
        ],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    collate_fn = TextAudioSpeakerCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=8,
    )

    optimizer = torch.optim.AdamW(
        t2s_model.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)

    for param_group in optimizer.param_groups:
        if 'initial_lr' not in param_group:
            param_group['initial_lr'] = hps.train.learning_rate

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=hps.train.lr_decay, last_epoch=epochs)

    if torch.cuda.is_available():
        t2s_model = DDP(t2s_model, device_ids=[rank], find_unused_parameters=False)
        semantic_model = DDP(semantic_model, device_ids=[rank], find_unused_parameters=False)
        semantic_codec = DDP(semantic_codec, device_ids=[rank], find_unused_parameters=False)
    else:
        t2s_model = t2s_model.to(device)
        semantic_model = semantic_model.to(device)
        semantic_codec = semantic_codec.to(device)

    t2s_model.train()
    semantic_model.eval()
    semantic_codec.eval()

    epoch_begin = keep_training_epochs
    print(epoch_begin)
            
    for epoch in range(epoch_begin, epochs + 1):
        train_epoch(hps,
                    rank,
                    epoch,
                    t2s_model,
                    optimizer,
                    train_loader,
                    maskgct_inference_pipeline)
        scheduler_g.step()

def train_epoch(hps, rank, epoch, t2s_model, optim, train_loader, maskgct_inference_pipeline):
    scaler = GradScaler(enabled=hps.train.fp16_run)
    global global_step
    accumulation_steps = hps.train.grad_accumulation_steps

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    optim = optim
    feature_criterion = nn.CrossEntropyLoss(reduction='none')
    train_loader.batch_sampler.set_epoch(epoch)
    for batch_idx, (x, x_lengths, y, y_lengths, sid) in enumerate(train_loader):

        if torch.cuda.is_available():
            x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(
                rank, non_blocking=True
            )
            y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(
                rank, non_blocking=True
            )
        else:
            x, x_lengths = x.to(device), x_lengths.to(device)
            y, y_lengths = y.to(device), y_lengths.to(device)

        with autocast(enabled=hps.train.fp16_run):
            with torch.no_grad():
                sematic_code, sematic_code_mask, phone_id, phone_id_mask = maskgct_inference_pipeline.maskgct_get_data_batch(y, y_lengths, x, x_lengths)
            logits, final_mask, x0, prompt_len, mask_prob = maskgct_inference_pipeline.t2s_model(sematic_code, sematic_code_mask, phone_id, phone_id_mask)
            logits = logits.transpose(1, 2)
        feature_loss = feature_criterion(logits, x0)
        final_mask = final_mask.squeeze(-1).float()
        masked_loss = feature_loss * final_mask
        final_loss = masked_loss.sum() / (final_mask.sum() + 1e-6)

        acc_loss = final_loss / accumulation_steps
        scaler.scale(acc_loss).backward()
        scaler.unscale_(optim)
        if (global_step + 1) % accumulation_steps == 0:
            scaler.step(optim)
            optim.zero_grad()
        scaler.update()
        
        if global_step % print_interval == 0 and rank == 0:
            print(f"Epoch: {epoch} [{100.0*batch_idx/len(train_loader):.0f}%]  Step: {global_step}  feature_loss:{final_loss:.4f}")

        global_step += 1

        if global_step % save_interval == 0 and rank == 0:
            print(f"saving: T2S_{epoch}_{global_step}.safetensors  ...")
            state_dict_new = t2s_model.state_dict()
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict_new.items()}
            safetensors.torch.save_file(new_state_dict, f'{hps.train.save_model_path}/T2S_{epoch}_{global_step}.safetensors')
            print("checkpoint saved.")


def main():
    assert torch.cuda.is_available(), "CPU training is not allowed."
    config_path = os.path.join('config', "train.json")
    hps = load_config(config_path)

    n_gpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(11451, 41919))

    mp.spawn(
        train_VC,
        nprocs=n_gpus,
        args=(
            n_gpus,
            hps,
        ),
    )

if __name__ == '__main__':
    main()
