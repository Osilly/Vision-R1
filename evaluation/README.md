# Evalution of Vision-R1

## Overview

We use [Evalscope](https://github.com/modelscope/evalscope) and vllm as the backend to evaluate Vision-R1.

Specifically, for Evalscope, we modified the model generation and answer extraction parts of the vlmeval backend it uses. For more details, please see our customized `evalscope/vlmeval`.

## Quick Start

### Install Dependencies

```bash
cd evalscope/
pip install -e '.[all]'
```

### Evaluation

#### Step1. Deploy an OpenAI API Service Using VLLM

```bash
MODEL_PATH="Vision-R1-7B"   # Replace with your model path
MODEL_NAME="Vision-R1-7B"
# Deploy example, you can also deploy model using multiple GPUs to accelerate inference
vllm serve ${MODEL_PATH} \
    --port 8000 \
    --host 0.0.0.0 \
    --dtype bfloat16 \
    --limit-mm-per-prompt image=5 \
    --served-model-name "${MODEL_NAME}" \
```

#### Step2. Run Evaluation

Please make sure you have a valid OPENAI API key to get the evaluation results normally.

We recommend using `temperature=0.6` and `top_p=0.95 `for best results.

```bash
cd evalscope/
# Replace with your openai keys
export OPENAI_API_BASE=https://api.xxx.com/v1/chat/completions
export OPENAI_API_KEY=xxx
# Run eval
python3 run_vision_r1_all/run_all.py \
	--work_dir outputs \
	--model_name Vision-R1-7B \
	--max_tokens 16000 \
	--temperature 0.6 \
	--top_p 0.95
```

Please note that Vision-R1 is a reasoning MLLM, thus the evaluation takes a long time. If all processes are completed normally, you can find all the benchmark results under `evalscope/${work_dir}`.
