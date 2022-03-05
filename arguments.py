import argparse


def create_arguments():
    parser = argparse.ArgumentParser(description="Arguments for training")
    parser.add_argument(
        "--train_dir",
        type=str,
        required=True,
        help="path to train directory containing text files"
    )
    parser.add_argument(
        "--train_csv_path",
        type=str,
        required=True,
        help="Path to train csv path containing the tags"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Hugging face model name or pretrained path. For example roberta-base etc"
    )
    parser.add_argument(
        "--max_len",
        type=int,
        required=True,
        help="Max length to be used by the model."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="Number of epochs to run. Number of steps will be calculated using this value"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Batch size for training"
    )
    parser.add_argument(
        "--max_learning_rate",
        type=float,
        required=True,
        help="Maximum learning rate for adam(default) optimizer. Initial learning rate will be calculated according to warmup_ratio parameter"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        required=True,
        help="Ratio of total steps for warmup. If value is 0.2 and total steps is 100, then first 0.2*100 = 20 will be used for warmup"
    )

    return parser
