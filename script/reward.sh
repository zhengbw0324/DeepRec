P=$1  # 5001
stage=$2  # cold or rec


export CUDA_VISIBLE_DEVICES=7

python -m server.reward \
  --port=$P \
  --log_file=${stage}.log \
  --stage=$stage

