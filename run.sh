while getopts  m:d:e:b:r:l:y:g:t:n:p: flag
do
    case "${flag}" in
        m) model_name_or_path=${OPTARG};;
        d) train_file=${OPTARG};;
        e) num_train_epochs=${OPTARG};;
        b) per_device_train_batch_size=${OPTARG};;
        r) learning_rate=${OPTARG};;
        l) max_seq_length=${OPTARG};;
        y) seed=${OPTARG};;
        g) NUM_GPU=${OPTARG};;
        t) eval_step=${OPTARG};;
        n) pretrained_model_name_or_path=${OPTARG};;
        p) phi=${OPTARG};;
    esac
done


file_name="data/${train_file}_for_simcse.csv"
f_name="result/sup-${train_file}-${model_name_or_path}-${seed}-${per_device_train_batch_size}-${phi}"

# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
PORT_ID=$(expr $RANDOM + 1000)

# Allow multiple threads    
export OMP_NUM_THREADS=8
# export TRANSFORMERS_OFFLINE=1

python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
--model_name_or_path $model_name_or_path --pretrained_model_name_or_path $pretrained_model_name_or_path  --train_file $file_name --output_dir $f_name \
--num_train_epochs $num_train_epochs --per_device_train_batch_size $per_device_train_batch_size \
--learning_rate $learning_rate --max_seq_length $max_seq_length --evaluation_strategy steps \
--metric_for_best_model stsb_spearman --load_best_model_at_end --eval_steps $eval_step --pooler_type cls \
--mlp_only_train --overwrite_output_dir --temp 0.05 --seed $seed --do_train --do_eval --phi $phi --fp16 
"$@"

python simcse_to_huggingface.py --path $f_name
python evaluation.py --model_name_or_path $f_name  --pooler cls  --task_set sts  --mode test
python evaluation.py --model_name_or_path $f_name  --pooler cls_before_pooler  --task_set sts  --mode test
