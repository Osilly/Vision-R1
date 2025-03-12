# MODEL should be same as $MODEL_NAME in `deploy_vllm.sh`
MODEL="Vision-R1"
TEMP=0.6
TOP_P=0.95
MAX_TOKENS=4096
IMAGE_PATH="./figs/example1.png"
PROMPT="Given a cone with a base radius represented by the variable 'r' (r = 1) and a slant height represented by the variable 's' (s = 3), determine the lateral surface area using variables.\nChoices:\nA: 2π\nB: 3π\nC: 6π\nD: 8π"
python3 vllm_inference.py \
    --model_path ${MODEL}  \
    --image_path ${IMAGE_PATH} \
    --prompt "${PROMPT}" \
    --max_tokens ${MAX_TOKENS} \
    --temperature ${TEMP} \
    --top_p ${TOP_P} \
