name=$1

model=./results/ckpts/${name}/_actor
config=Qwen/Qwen2.5-7B


set -x

for step in `ls ${model} | grep global_step | awk -F'_' '{ print $2 }'`
do
mkdir ${model}/checkpoint_${step}

python ${model}/zero_to_fp32.py \
              ${model} \
              ${model}/checkpoint_${step}/pytorch_model.bin \
              -t global_${step}

cp ${config}/*.json ${model}/checkpoint_${step}/
rm -rf ${model}/checkpoint_${step}/*.index.json
python ./script/save_ckpt.py -s ${model}/checkpoint_${step} -t ${model}/checkpoint_${step}
done