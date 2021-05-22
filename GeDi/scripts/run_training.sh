

export lambda_=0.80
export lr=2e-5

#final GeDi LM checkpoint saved at --output_dir
python ../train_GeDi.py --task_name SST-2 \
  --output_dir "/home/memesrized/projects/education/diploma/GeDi/pretrained_models/custom/" \
  --overwrite_output_dir \
  --no_cuda \
  --do_train \
  --logit_scale \
  --data_dir "/home/memesrized/projects/education/diploma/GeDi/data/"  \
  --max_seq_length 512 \
  --overwrite_cache \
  --per_gpu_train_batch_size 1 \
  --per_gpu_eval_batch_size  1 \
  --learning_rate $lr  \
  --num_train_epochs 1.0  \
  --model_type gpt2  \
  --model_name_or_path "/home/memesrized/projects/education/diploma/models/140/" \
  --gen_weight $lambda_ \
  --logging_steps 500 \
  --save_steps 5000000000 \
  --code_0 "dirty" \
  --code_1 "clean" \
