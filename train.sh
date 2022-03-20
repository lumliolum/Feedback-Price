python arguments.py \
    --train_dir "../../input/feedback-prize-2021/train" \
    --train_csv_path "../../input/feedback-prize-2021/train.csv" \
    --test_dir "../../input/feedback-prize-2021/test" \
    --model_name_or_path "roberta-base" \
    --max_len 512 \
    --epochs 5 \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --max_learning_rate 1e-5 \
    --use_scheduler 1 \
    --warmup_ratio 0.2 \
    --output_dir "checkpoint"
