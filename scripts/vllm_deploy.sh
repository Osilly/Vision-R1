MODEL_PATH="Vision-R1-7B"   # Replace with your model path
MODEL_NAME="Vision-R1-7B"
# deploy
vllm serve ${MODEL_PATH} \
    --port 8000 \
    --host 0.0.0.0 \
    --dtype bfloat16 \
    --limit-mm-per-prompt image=5 \
    --served-model-name "${MODEL_NAME}" \
