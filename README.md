# AurA for Toxicity Mitigation - MLLM Revision

<p align="center">
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>
The multimodal MLLM implementation of https://github.com/apple/ml-aura, any use must be authorized by xsuaucuadros@apple.com.

## Getting Started 
Recommend GPU - `A100 PCLe 80G * 1`

### 1. Clone this repository

```bash
git clone https://github.com/dut-laowang/ml-aura-vl.git
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
### 3. Download the dataset and store in the pics/ pics_500/

Download the dataset

`pics/` : [https://drive.google.com/drive/folders/1rxrxYQXQw-0Q9BM5PriVV87F3lDsk87X?usp=sharing](https://drive.google.com/file/d/1Tr8X-uFM6czqo8UvlmklieGDXvCr-ARf/view?usp=sharing)

`pics_500/` : [https://drive.google.com/drive/folders/1zpjc662970w1cJLPXGXPmvWq_66lJ-SE?usp=sharing](https://drive.google.com/file/d/1plmj-HPWsKY1O9n0jxY3Ocu_ap9dm7bv/view?usp=sharing)
- `pics/` together with `train.xlsx` for `1. Extract Responses` && `2. Compute AURA intervention`
- `pics_500/` together with `test.xlsx` for `3. Generate with intervened model and Test`
```bash
> ls ml-aura-vl/pics
animal_abuse__0__0.jpg
...
```
```bash
> ls ml-aura-vl/llava1
train.xlsx
test.xlsx
...
```

## Usage

Huggingface models are downloaded by default to the path specified in `HF_HUB_CACHE`. For more information visit the official Huggingface website.

### 1. Extract Responses

```bash
python -m scripts.llava_response \
  --model-path llava-hf/llava-1.5-7b-hf \
  --dataset llava1 \
  --data-dir /workspace/ml-aura-vl \
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
  --data-dir /workspace/ml-aura-vl \
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

The generation will be the form as `generated_outputs_with_hooks.csv`

