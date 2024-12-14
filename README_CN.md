## MaskGCT: Zero-Shot Text-to-Speech with Masked Generative Codec Transformer  
本项目基于开源项目 [MaskGCT](https://github.com/open-mmlab/Amphion)，实现了T2S模型的训练部分，可以实现新语种的训练或自定义T2S模型等任务。
## 使用方法
### 安装

```
git clone https://github.com/nene1212/MaskGCT-Training.git
sudo apt-get install espeak-ng

conda create -n maskgct python=3.10
conda activate maskgct
pip install -r requirements.txt
```
### 准备预训练模型

| Model Name                                                   | Description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Semantic Codec](https://huggingface.co/amphion/MaskGCT/tree/main/semantic_codec) | Converting speech to semantic tokens.                        |
| [Acoustic Codec](https://huggingface.co/amphion/MaskGCT/tree/main/acoustic_codec) | Converting speech to acoustic tokens and reconstructing waveform from acoustic tokens. |
| [MaskGCT-T2S](https://huggingface.co/amphion/MaskGCT/tree/main/t2s_model) | Predicting semantic tokens with text and prompt semantic tokens. |
| [MaskGCT-S2A](https://huggingface.co/amphion/MaskGCT/tree/main/s2a_model) | Predicts acoustic tokens conditioned on semantic tokens.     |
| [w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0) | Wav2Vec2BertModel.                                           |


将以上模型存放到`./MaskGCT_model/`的对应文件夹中。

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

### 准备数据集
参考[VITS](https://github.com/jaywalnut310/vits)项目中filelist的生成，准备`path_to_wav|spk_id|text`格式的txt文件`filelists/filelist.txt`用来加载数据集。
```
/vctk/p234/p234_112.wav|3|That would be a serious problem.
/vctk/p298/p298_125.wav|68|I asked why he had come.
/vctk/p229/p229_128.wav|67|The whole process is a vicious circle at the moment.
/vctk/p283/p283_318.wav|95|If not, he should go home.
/vctk/p260/p260_046.wav|81|It is marvellous.
```
### 开始训练

```
python train.py
```

### 训练参数
你也可以通过调整train.json以及maskgct.json来调整基本的训练参数。

| Param(`train.json`)     | Description                                      |
| ----------------------- | ------------------------------------------------ |
| log_interval            | 每隔多少step打印一次                             |
| epochs                  | 训练的总Epoch数                                  |
| learning_rate           | 学习率                                           |
| grad_accumulation_steps | 梯度累积次数                                     |
| save_interval           | 每多少step保存一次                               |
| keep_training_epochs    | 继续上一次训练的epoch，设置为-1时使用官方T2S模型 |
| keep_training_steps     | 继续上一次训练的step                             |
## References
[MaskGCT](https://github.com/open-mmlab/Amphion)
[VITS](https://github.com/jaywalnut310/vits)

