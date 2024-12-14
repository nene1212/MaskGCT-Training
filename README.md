## MaskGCT: Zero-Shot Text-to-Speech with Masked Generative Codec Transformer  
This project is based on the open-source project [MaskGCT](https://github.com/open-mmlab/Amphion) and implements the training part of the T2S model. It supports training for new languages or customizing T2S models for various tasks.

## Usage
### Installation

```
git clone https://github.com/nene1212/MaskGCT-Training.git
sudo apt-get install espeak-ng

conda create -n maskgct python=3.10
conda activate maskgct
pip install -r requirements.txt
```
### Prepare Pre-trained Models

| Model Name                                                                        | Description                                                                            |
| --------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| [Semantic Codec](https://huggingface.co/amphion/MaskGCT/tree/main/semantic_codec) | Converting speech to semantic tokens.                                                  |
| [Acoustic Codec](https://huggingface.co/amphion/MaskGCT/tree/main/acoustic_codec) | Converting speech to acoustic tokens and reconstructing waveform from acoustic tokens. |
| [MaskGCT-T2S](https://huggingface.co/amphion/MaskGCT/tree/main/t2s_model)         | Predicting semantic tokens with text and prompt semantic tokens.                       |
| [MaskGCT-S2A](https://huggingface.co/amphion/MaskGCT/tree/main/s2a_model)         | Predicts acoustic tokens conditioned on semantic tokens.                               |
| [w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0)                      | Wav2Vec2BertModel.                                                                     |

Store the models in the corresponding folders under `./MaskGCT_model/`.
```
./MaskGCT_model
├── acoustic_codec
│   ├── model_1.safetensors
│   └── model.safetensors
├── s2a_model
│   ├── s2a_model_1layer
│   │   └── model.safetensors
│   └── s2a_model_full
│       └── model.safetensors
├── semantic_codec
│   └── model.safetensors
├── t2s_model
│   └── model.safetensors
└── w2v_bert
    ├── config.json
    ├── conformer_shaw.pt
    ├── model.safetensors
    └── preprocessor_config.json
```
### Prepare Dataset
Refer to the filelist in the [VITS](https://github.com/jaywalnut310/vits) project and prepare a `path_to_wav|spk_id|text` formatted text file named `filelists/filelist.txt` to load the dataset.
```
/vctk/p234/p234_112.wav|3|That would be a serious problem.
/vctk/p298/p298_125.wav|68|I asked why he had come.
/vctk/p229/p229_128.wav|67|The whole process is a vicious circle at the moment.
/vctk/p283/p283_318.wav|95|If not, he should go home.
/vctk/p260/p260_046.wav|81|It is marvellous.
```
### Training
```
python train.py
```

### Training Parameters
You can adjust training parameters by modifying `train.json` and `maskgct.json`.

| Param(`train.json`)     | Description                                                                |
| ----------------------- | -------------------------------------------------------------------------- |
| log_interval            | Print logs every N steps                                                   |
| epochs                  | Total number of epochs                                                     |
| grad_accumulation_steps | Gradient accumulation steps                                                |
| save_interval           | Save model every N steps                                                   |
| keep_training_epochs    | Continue training from epoch N. When N=-1, the official T2S model is used. |
| keep_training_steps     | Continue training from step N                                              |

## References
[MaskGCT](https://github.com/open-mmlab/Amphion)  
[VITS](https://github.com/jaywalnut310/vits)
