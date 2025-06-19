# AurA for Toxicity Mitigation - MLLM Revision

<p align="center">
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>
The multimodal MLLM implementation of https://github.com/apple/ml-aura, any use must be authorized by xsuaucuadros@apple.com.

## Getting Started 

### 1. Clone this repository

```bash
git clone https://github.com/dutlao-wang/ml-aura-vl.git
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

Optionally install this repository

```bash
pip install -e .
```
```bash
export PYTHONPATH=.
```
```bash
pip install -U transformers
```
```bash
pip install openpyxl
```
### 3. Download the dataset and store in the pics/ pics/500


```bash
> ls $DATA_DIR/llava1
train.xlsx
test.xlsx
...
```

## Usage

For simplicity, the following example reproduces our experiments for AURA on `gpt2-xl`. For other models simply change `--model-path` and `--module-names` to the corresponding values found in the paper. Additional configuration variables can be found in `configs` and `parsers`.

Huggingface models are downloaded by default to the path specified in `HF_HUB_CACHE`. For more information visit the official Huggingface website.

### 1. Extract Responses

```bash
python -m scripts.llava_response \
  --model-path llava-hf/llava-1.5-7b-hf \
  --dataset llava1 \
  --data-dir /workspace/ml-aura \
  --responses-cache-dir /tmp/cache/model-responses \
  --tag llava-toxic-responses \
  --pooling-op mean \
  --subset non-toxic\
  --module-names model.language_model.layers.*.mlp.up_proj model.language_model.layers.*.mlp.gate_proj model.language_model.layers.*.mlp.down_proj

```
```bash
python -m scripts.llava_response \
  --model-path llava-hf/llava-1.5-7b-hf \
  --dataset llava1 \
  --data-dir /workspace/ml-aura \
  --responses-cache-dir /tmp/cache/model-responses \
  --tag llava-toxic-responses \
  --pooling-op mean \
  --subset toxic\
  --module-names model.language_model.layers.*.mlp.up_proj model.language_model.layers.*.mlp.gate_proj model.language_model.layers.*.mlp.down_proj

```
The output will be written in the following folder structure:

```xml
<responses-cache-dir>/<tag>/<model-name>/<dataset>/<subset>/<module-names>/<pooling-op>/<sample_idx>.pt
```

By default `args.responses-cache-dir` is set to `/tmp/cache`.

### 2. Compute AURA intervention

Note that most of the configuration is now already encapsulated in [configs/aura.yaml](configs/aura.yaml).

```bash
python -m scripts.learn_aura \
  --config-path configs/llava_aura.yaml \
  --module-names 'model.language_model.layers.*.mlp.up_proj' 'model.language_model.layers.*.mlp.gate_proj' 'model.language_model.layers.*.mlp.down_proj'

```

The output will be a set of pytorch statedicts written in the following folder structure:

```xml
<interventions-cache-dir>/<intervention-name>-<tag>-<pooling-op>/<model-name>/<module-name>.statedict
```

By default `args.interventions-cache-dir` is set to `/tmp/cache/model-interventions`

### 3. Generate with intervened model and Test

```bash
python -m scripts.generate_with_hooks_llava_500_prompt \
  --model-path llava-hf/llava-1.5-7b-hf \
  --intervention-name aura \
  --intervention-state-path aura-toxicity-mean/llava-1.5-7b-hf \
  --module-names 'model.language_model.layers.*.mlp.up_proj' 'model.language_model.layers.*.mlp.gate_proj' 'model.language_model.layers.*.mlp.down_proj' \
  --device cuda \
  --verbose 1
```


## Citation
```bibtex
@inproceedings{
suau2024whispering,
title={Whispering Experts: Neural Interventions for Toxicity Mitigation in Language Models},
author={Xavier Suau and Pieter Delobelle and Katherine Metcalf and Armand Joulin and Nicholas Apostoloff and Luca Zappella and Pau Rodriguez},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=2P6GVfSrfZ}
}
```
