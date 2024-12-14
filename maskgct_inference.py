# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from maskgct_utils import *
import safetensors
import soundfile as sf
import time

t2s_model_epoch = -1
t2s_model_step = 0

if __name__ == "__main__":

    # build model
    device = torch.device("cuda:0")
    #device = torch.device("cpu")
    cfg_path = "./config/maskgct.json"
    cfg = load_config(cfg_path)
    # 1. build semantic model (w2v-bert-2.0)
    semantic_model, semantic_mean, semantic_std = build_semantic_model(device)
    # 2. build semantic codec
    semantic_codec = build_semantic_codec(cfg.model.semantic_codec, device)
    # 3. build acoustic codec
    codec_encoder, codec_decoder = build_acoustic_codec(
        cfg.model.acoustic_codec, device
    )
    # 4. build t2s model
    t2s_model = build_t2s_model(cfg.model.t2s_model, device)
    # 5. build s2a model
    s2a_model_1layer = build_s2a_model(cfg.model.s2a_model.s2a_1layer, device)
    s2a_model_full = build_s2a_model(cfg.model.s2a_model.s2a_full, device)

    semantic_code_ckpt = r'./MaskGCT_model/semantic_codec/model.safetensors'
    codec_encoder_ckpt = r'./MaskGCT_model/acoustic_codec/model.safetensors'
    codec_decoder_ckpt = r'./MaskGCT_model/acoustic_codec/model_1.safetensors'
    s2a_1layer_ckpt = r'./MaskGCT_model/s2a_model/s2a_model_1layer/model.safetensors'
    s2a_full_ckpt = r'./MaskGCT_model/s2a_model/s2a_model_full/model.safetensors'

    if t2s_model_epoch == -1:
        t2s_model_ckpt = r'./MaskGCT_model/t2s_model/model.safetensors'
    else:
        t2s_model_ckpt = f'./ckpt/T2S_{t2s_model_epoch}_{t2s_model_step}.safetensors'

    # load semantic codec
    safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
    # load acoustic codec
    safetensors.torch.load_model(codec_encoder, codec_encoder_ckpt)
    safetensors.torch.load_model(codec_decoder, codec_decoder_ckpt)
    # load t2s model
    safetensors.torch.load_model(t2s_model, t2s_model_ckpt)
    # load s2a model
    safetensors.torch.load_model(s2a_model_1layer, s2a_1layer_ckpt)
    safetensors.torch.load_model(s2a_model_full, s2a_full_ckpt)

    # inference
    prompt_wav_path = "./wav/prompt.wav"
    prompt_text = " And it is worth mention in passing that, as an example of fine typography."

    target_text = []
    target_text.append(' It incorporates advanced feature extraction methods to handle audio encoding, ')
    target_text.append(' Prosody preservation across languages and modalities.')

    config_path = os.path.join('config', "train.json")
    hps = load_config(config_path)

    # Specify the target duration (in seconds). If target_len = None, we use a simple rule to predict the target duration.
    target_len = 6
    maskgct_inference_pipeline = MaskGCT_Inference_Pipeline(
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
        hps,
    )
    for i in range(len(target_text)):
        start_time = time.time()
        recovered_audio = maskgct_inference_pipeline.maskgct_inference(
            prompt_wav_path, prompt_text, target_text[i], "EN", "EN", target_len=target_len, n_timesteps=25
        )
        end_time = time.time()
        sf.write(f'./wav/converted_{i}.wav', recovered_audio, 24000)
        print(f'finished_{i}:{end_time-start_time:.2f}')
