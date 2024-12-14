# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import os
import pickle
import math
import json
import accelerate
import safetensors
from utils.util import load_config
from tqdm import tqdm

from codec.kmeans.repcodec_model import RepCodec
from maskgct_s2a import MaskGCT_S2A
from maskgct_t2s import MaskGCT_T2S
from codec.amphion_codec.codec import CodecEncoder, CodecDecoder
from transformers import Wav2Vec2BertModel

from g2p.g2p_generation import g2p, chn_eng_g2p, p2i

from transformers import SeamlessM4TFeatureExtractor

from module import commons

import logging
logging.getLogger("phonemizer").setLevel(logging.ERROR)


def g2p_(text, language):
    if language in ["zh", "en"]:
        return chn_eng_g2p(text)
    else:
        return g2p(text, sentence=None, language=language)

def p2i_(text, language):
    if language in ["zh", "en"]:
        return chn_eng_g2p(text)
    else:
        return p2i(text, sentence=None, language=language)


def build_t2s_model(cfg, device):
    t2s_model = MaskGCT_T2S(cfg=cfg)
    t2s_model.eval()
    t2s_model.to(device)
    return t2s_model


def build_s2a_model(cfg, device):
    soundstorm_model = MaskGCT_S2A(cfg=cfg)
    soundstorm_model.eval()
    soundstorm_model.to(device)
    return soundstorm_model


def build_semantic_model(device):
    #semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
    semantic_model = Wav2Vec2BertModel.from_pretrained("./MaskGCT_model/w2v_bert/")
    semantic_model.eval()
    semantic_model.to(device)
    stat_mean_var = torch.load("./ckpt/wav2vec2bert_stats.pt")
    semantic_mean = stat_mean_var["mean"]
    semantic_std = torch.sqrt(stat_mean_var["var"])
    semantic_mean = semantic_mean.to(device)
    semantic_std = semantic_std.to(device)
    return semantic_model, semantic_mean, semantic_std


def build_semantic_codec(cfg, device):
    semantic_codec = RepCodec(cfg=cfg)
    semantic_codec.eval()
    semantic_codec.to(device)
    return semantic_codec


def build_acoustic_codec(cfg, device):
    codec_encoder = CodecEncoder(cfg=cfg.encoder)
    codec_decoder = CodecDecoder(cfg=cfg.decoder)
    codec_encoder.eval()
    codec_decoder.eval()
    codec_encoder.to(device)
    codec_decoder.to(device)
    return codec_encoder, codec_decoder


class MaskGCT_Inference_Pipeline:
    def __init__(
        self,
        semantic_model,
        semantic_codec,
        codec_encoder,
        codec_decoder,
        t2s_model,
        s2a_model_1layer,
        s2a_model_full,
        semantic_mean,
        semantic_std,
        device,
        hps=None,
    ):
        self.processor = SeamlessM4TFeatureExtractor.from_pretrained(
            "./MaskGCT_model/w2v_bert/"
        )
        self.semantic_model = semantic_model
        self.semantic_codec = semantic_codec
        self.codec_encoder = codec_encoder
        self.codec_decoder = codec_decoder
        self.t2s_model = t2s_model
        self.s2a_model_1layer = s2a_model_1layer
        self.s2a_model_full = s2a_model_full
        self.semantic_mean = semantic_mean
        self.semantic_std = semantic_std
        self.device = device
        self.hps = hps

    @torch.no_grad()
    def extract_features(self, speech):
        inputs = self.processor(speech, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"][0]
        attention_mask = inputs["attention_mask"][0]
        return input_features, attention_mask

    @torch.no_grad()
    def extract_semantic_code(self, input_features, attention_mask):
        vq_emb = self.semantic_model(           # Wav2Vec2BertModel
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[17]  # (B, T, C)
        feat = (feat - self.semantic_mean.to(feat)) / self.semantic_std.to(feat)

        semantic_code, rec_feat = self.semantic_codec.quantize(feat)  # (B, T)
        return semantic_code, rec_feat

    @torch.no_grad()
    def extract_acoustic_code(self, speech):
        vq_emb = self.codec_encoder(speech.unsqueeze(1))
        _, vq, _, _, _ = self.codec_decoder.quantizer(vq_emb)
        acoustic_code = vq.permute(1, 2, 0)
        return acoustic_code

    @torch.no_grad()
    def text2semantic(
        self,
        prompt_speech,
        prompt_text,
        prompt_language,
        target_text,
        target_language,
        target_len=None,
        n_timesteps=50,
        cfg=2.5,
        rescale_cfg=0.75,
    ):
        prompt_phone_id = g2p_(prompt_text, prompt_language)[1]
        target_phone_id = g2p_(target_text, target_language)[1]

        if target_len is None:
            target_len = int(
                (len(prompt_speech) * len(target_phone_id) / len(prompt_phone_id))
                / 16000
                * 50
            )
        else:
            target_len = int(target_len * 50)

        prompt_phone_id = torch.tensor(prompt_phone_id, dtype=torch.long).to(
            self.device
        )
        target_phone_id = torch.tensor(target_phone_id, dtype=torch.long).to(
            self.device
        )

        phone_id = torch.cat([prompt_phone_id, target_phone_id])

        input_features, attention_mask = self.extract_features(prompt_speech)
        input_features = input_features.unsqueeze(0).to(self.device)
        attention_mask = attention_mask.unsqueeze(0).to(self.device)
        semantic_code, _ = self.extract_semantic_code(input_features, attention_mask)

        
        predict_semantic = self.t2s_model.reverse_diffusion(
            semantic_code[:, :],
            target_len,
            phone_id.unsqueeze(0),
            n_timesteps=n_timesteps,
            cfg=cfg,
            rescale_cfg=rescale_cfg,
        )

        print("predict semantic shape", predict_semantic.shape)

        combine_semantic_code = torch.cat(
            [semantic_code[:, :], predict_semantic], dim=-1
        )
        prompt_semantic_code = semantic_code

        return combine_semantic_code, prompt_semantic_code

    @torch.no_grad()
    def semantic2acoustic(
        self,
        combine_semantic_code,
        acoustic_code,
        n_timesteps=[25, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        cfg=2.5,
        rescale_cfg=0.75,
    ):
        semantic_code = combine_semantic_code

        cond = self.s2a_model_1layer.cond_emb(semantic_code)
        prompt = acoustic_code[:, :, :]
        predict_1layer = self.s2a_model_1layer.reverse_diffusion(
            cond=cond,
            prompt=prompt,
            temp=1.5,
            filter_thres=0.98,
            n_timesteps=n_timesteps[:1],
            cfg=cfg,
            rescale_cfg=rescale_cfg,
        )

        cond = self.s2a_model_full.cond_emb(semantic_code)
        prompt = acoustic_code[:, :, :]
        predict_full = self.s2a_model_full.reverse_diffusion(
            cond=cond,
            prompt=prompt,
            temp=1.5,
            filter_thres=0.98,
            n_timesteps=n_timesteps,
            cfg=cfg,
            rescale_cfg=rescale_cfg,
            gt_code=predict_1layer,
        )

        vq_emb = self.codec_decoder.vq2emb(
            predict_full.permute(2, 0, 1), n_quantizers=12
        )
        recovered_audio = self.codec_decoder(vq_emb)
        prompt_vq_emb = self.codec_decoder.vq2emb(
            prompt.permute(2, 0, 1), n_quantizers=12
        )
        recovered_prompt_audio = self.codec_decoder(prompt_vq_emb)
        recovered_prompt_audio = recovered_prompt_audio[0][0].cpu().numpy()
        recovered_audio = recovered_audio[0][0].cpu().numpy()
        combine_audio = np.concatenate([recovered_prompt_audio, recovered_audio])

        return combine_audio, recovered_audio

    def maskgct_inference(
        self,
        prompt_speech_path,
        prompt_text,
        target_text,
        language="en",
        target_language="en",
        target_len=None,
        n_timesteps=25,
        cfg=2.5,
        rescale_cfg=0.75,
        n_timesteps_s2a=[25, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        cfg_s2a=2.5,
        rescale_cfg_s2a=0.75,
    ):
        speech_16k = librosa.load(prompt_speech_path, sr=16000)[0]
        speech = librosa.load(prompt_speech_path, sr=24000)[0]

        combine_semantic_code, _ = self.text2semantic(
            speech_16k,
            prompt_text,
            language,
            target_text,
            target_language,
            target_len,
            n_timesteps,
            cfg,
            rescale_cfg,
        )

        acoustic_code = self.extract_acoustic_code(
            torch.tensor(speech).unsqueeze(0).to(self.device)
        )
        _, recovered_audio = self.semantic2acoustic(
            combine_semantic_code,
            acoustic_code,
            n_timesteps=n_timesteps_s2a,
            cfg=cfg_s2a,
            rescale_cfg=rescale_cfg_s2a,
        )

        return recovered_audio

    def maskgct_get_data_batch_v2(self, prompt_speech, prompt_speech_len, target_speech, target_speech_len, prompt_text, prompt_text_len, target_text, target_text_len):
        prompt_phone_id = prompt_text
        target_phone_id = target_text

        batch_size, max_len = prompt_phone_id.shape
        range_tensor = torch.arange(max_len, device=prompt_phone_id.device).unsqueeze(0).expand(batch_size, max_len)
        prompt_text_mask = range_tensor < prompt_text_len.unsqueeze(1)
        batch_size, max_len = target_phone_id.shape
        range_tensor = torch.arange(max_len, device=target_phone_id.device).unsqueeze(0).expand(batch_size, max_len)
        target_text_mask = range_tensor < target_text_len.unsqueeze(1)

        phone_id = torch.cat([prompt_phone_id, target_phone_id], dim=1)
        phone_id_mask = torch.cat([prompt_text_mask, target_text_mask], dim=1)

        input_features, input_attention_mask = self.extract_features_batch(
            prompt_speech, prompt_speech_len)  # SeamlessM4TFeatureExtractor将wav转换为[len,160]的语音表示向量
        input_semantic_code, _ = self.extract_semantic_code(input_features, input_attention_mask)  # 将语音向量通过Wav2Vec2BertModel + RVQ转换为sematic code

        target_features, target_attention_mask = self.extract_features_batch(
            target_speech, target_speech_len)
        target_semantic_code, _ = self.extract_semantic_code(target_features, target_attention_mask)

        sematic_code = torch.zeros_like(torch.cat([input_semantic_code, target_semantic_code], dim=1))
        sematic_code_mask = torch.zeros_like(torch.cat([input_attention_mask, target_attention_mask], dim=1))

        for i in range(input_features.size(0)):
            input_semantic_len = torch.sum(input_attention_mask[i] == 1).item()
            target_semantic_len = torch.sum(target_attention_mask[i] == 1).item()
            sematic_code[i, :(input_semantic_len+target_semantic_len)] = torch.cat((input_semantic_code[i, :input_semantic_len], target_semantic_code[i, :target_semantic_len]), dim=0)
            sematic_code_mask[i, :(input_semantic_len + target_semantic_len)] = 1

        return sematic_code, sematic_code_mask, phone_id, phone_id_mask

    def maskgct_get_data_batch(self, speech, speech_len, text, text_len):
        phone_id = text
        batch_size, max_len = phone_id.shape
        range_tensor = torch.arange(max_len, device=phone_id.device).unsqueeze(0).expand(batch_size, max_len)
        phone_id_mask = range_tensor < text_len.unsqueeze(1)

        features, attention_mask = self.extract_features_batch(
            speech, speech_len)  # SeamlessM4TFeatureExtractor将wav转换为[len,160]的语音表示向量
        semantic_code, _ = self.extract_semantic_code(features, attention_mask)  # 将语音向量通过Wav2Vec2BertModel + RVQ转换为sematic code

        return semantic_code, attention_mask, phone_id, phone_id_mask

    def maskgct_get_data(self, prompt_speech, prompt_speech_len, target_speech, target_speech_len, prompt_text, prompt_text_len, target_text, target_text_len):
        prompt_phone_id = prompt_text
        target_phone_id = target_text

        prompt_phone_id = prompt_phone_id[:, :prompt_text_len[0]]  # 截取有效部分
        target_phone_id = target_phone_id[:, :target_text_len[0]]

        phone_id = torch.cat([prompt_phone_id, target_phone_id], dim=1)
        phone_id_mask = torch.ones_like(phone_id, dtype=torch.bool)

        device = self.device
        prompt_speech_cpu = prompt_speech[0, :prompt_speech_len[0]].cpu()
        target_speech_cpu = target_speech[0, :target_speech_len[0]].cpu()

        input_features, input_attention_mask = self.extract_features(prompt_speech_cpu)
        input_features = input_features.unsqueeze(0).to(device)
        input_attention_mask = input_attention_mask.unsqueeze(0).to(device)
        input_semantic_code, _ = self.extract_semantic_code(input_features, input_attention_mask)

        target_features, target_attention_mask = self.extract_features(target_speech_cpu)
        target_features = target_features.unsqueeze(0).to(device)
        target_attention_mask = target_attention_mask.unsqueeze(0).to(device)
        target_semantic_code, _ = self.extract_semantic_code(target_features, target_attention_mask)

        sematic_code = torch.cat([input_semantic_code, target_semantic_code], dim=1)
        sematic_code_mask = torch.cat([input_attention_mask, target_attention_mask], dim=1)

        return sematic_code, sematic_code_mask, phone_id, phone_id_mask


    def extract_features_batch(self, speech, speech_len):
        batch_size = speech.shape[0]
        device = self.device

        prompt_speech_cpu = speech[0, :speech_len[0]].cpu()
        input_features_sample, input_attention_mask_sample = self.extract_features(prompt_speech_cpu)

        all_input_features = torch.zeros((batch_size, *input_features_sample.shape),
                                         device=device)  # [batch_size, len, 160]
        all_input_attention_masks = torch.zeros((batch_size, *input_attention_mask_sample.shape),
                                                device=device)  # [batch_size, len]

        for i in range(batch_size):
            prompt_speech_cpu = speech[i, :, :speech_len[i]].cpu()
            input_features, input_attention_mask = self.extract_features(prompt_speech_cpu)
            all_input_features[i, :input_features.shape[0], :] = input_features.to(device)
            all_input_attention_masks[i, :input_attention_mask.shape[0]] = input_attention_mask.to(device)

        return all_input_features, all_input_attention_masks
