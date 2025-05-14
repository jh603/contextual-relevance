start_vllm_instance() {
    GPU_ID=$1
    PORT=$2
    MODEL_PATH=$3

    echo "Starting vLLM instance on GPU $GPU_ID at port $PORT with model: $MODEL_PATH..."

    BASE_CMD="CUDA_VISIBLE_DEVICES=$GPU_ID python -m vllm.entrypoints.openai.api_server \
        --model \"$MODEL_PATH\" \
        --host \"127.0.0.1\" \
        --port \"$PORT\" \
        --pipeline-parallel-size 1 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.9 \
        --max-num-seqs 10 \
        --disable-log-stats \
        --enforce-eager"

    if [[ "$MODEL_PATH" == *"Ministral"* ]] || [[ "$MODEL_PATH" == *"mistral"* ]]; then
        BASE_CMD="$BASE_CMD \
        --tokenizer_mode mistral \
        --config_format mistral \
        --load_format mistral"
    fi

    eval "$BASE_CMD &"

    echo "vLLM instance started on GPU $GPU_ID at port $PORT."
}

MODEL="Qwen/Qwen2.5-7B-Instruct"

start_vllm_instance 0 5000 "$MODEL"

wait

echo "All vLLM instances have been started."
