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
### 3. Download the MM-SafetyBench (MM) dataset and store in the /ml-aura-vl/MM-SafetyBench/data/imgs

Download the MM dataset : [https://drive.google.com/drive/folders/1rxrxYQXQw-0Q9BM5PriVV87F3lDsk87X?usp=sharing](https://drive.google.com/file/d/1Tr8X-uFM6czqo8UvlmklieGDXvCr-ARf/view?usp=sharing)

Run```python rename.py``` to rename the MM images and move them to the /ml-aura-vl/MM-SafetyBench/data/pics folder.

```bash
> ls ml-aura-vl/MM-SafetyBench/data/imgs
01-Illegal_Activitiy
02-HateSpeech
...
```
```bash
> ls ml-aura-vl/MM-SafetyBench/data/pics
01-Illegal_Activitiy_0.jpg
01-Illegal_Activitiy_1.jpg
...
```

## Usage

Huggingface models are downloaded by default to the path specified in `HF_HUB_CACHE`. For more information visit the official Huggingface website.

### 1. Extract Responses (Twice: toxic/non-toxic)

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
python -m scripts.generate_with_hooks_mmbench_llava \
  --model-path llava-hf/llava-1.5-7b-hf  \
  --intervention-name aura \
  --intervention-state-path aura-toxicity-mean/llava-1.5-7b-hf \
  --module-names 'model.language_model.layers.*.mlp.up_proj' 'model.language_model.layers.*.mlp.gate_proj' 'model.language_model.layers.*.mlp.down_proj' \
  --device cuda \
  --dtype float32 \
  --verbose 1
```

Run the MM-SafetyBench/evaluation.py and can see the result in ```\eval_results```

