python train.py \
    --train_dir "../../input/feedback-prize-2021/train/" \
    --train_csv_path "../../input/feedback-prize-2021/train.csv" \
    --model_name_or_path "roberta-base" \
    --max_len 512 \
    --epochs 5 \
    --batch_size 8 \
    --max_learning_rate 1e-4 \
    --warmup_ratio 0.2 \
