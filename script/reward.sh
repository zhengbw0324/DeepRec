P=$1  # 6001
stage=$2  # cold or rec


export CUDA_VISIBLE_DEVICES=7


run_name="${stage}"
echo "run_name: ${run_name}"
python -m server.reward \
  --config_file=./server/config/Reward.yaml \
  --port=$P \
  --log_file=${stage}.log \
  --stage=$stage

