
gpu=$1
start=$2
end=$3
CKPT=$4

echo  "start: $start, end: $end"
echo  "gpu: $gpu"


data_name=Video_Games
DATA_PATH=./data/dataset/${data_name}/RL/train.jsonl
MODEL_CKPT=./results/ckpts/
OUTPUT_DIR=./results/eval/


recall_port=6001
SERVER_IP=XXXXXXXX
RECALL_SERVER=http://${SERVER_IP}:${recall_port}/recall


STEPS=(80)

for STEP in ${STEPS[@]}; do

  MODEL_PATH=${MODEL_CKPT}${CKPT}/_actor/checkpoint_step${STEP}

  python -m evaluation.eval_rec \
    --data_file $DATA_PATH \
    --recall_server $RECALL_SERVER \
    --gpu_id $gpu \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --start_data_idx $start \
    --end_data_idx $end \
    --rec_topk 10 \
    --recall_topk 20 \
    --chunk_size 2000 \
    --prompt_type "base"

done




