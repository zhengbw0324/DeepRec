
topk=$1
gpu=$2
start=$3
end=$4

echo  "start: $start, end: $end"
echo  "gpu: $gpu"
echo  "topk: $topk"

#sleep 600

DATA_PATH=XXXXX
MODEL_CKPT=./results/ckpts/
OUTPUT_DIR=./results/eval/


recall_port=XXXX
SERVER_IP=XXXXXXXX
RECALL_SERVER=http://${SERVER_IP}:${recall_port}/recall


CKPTS=(
"XXXXXX"
)
STEPS=(80)

for CKPT in ${CKPTS[@]}; do
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
      --recall_topk $topk \
      --chunk_size 2000 \
      --prompt_type "base"

  done
done




