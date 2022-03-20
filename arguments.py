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
        "--test_dir",
        type=str,
        default=None,
        help="path to test directory containing text files for which prediction will be run"
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
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps to use. if you dont want to use, make the value to 1"
    )
    parser.add_argument(
        "--max_learning_rate",
        type=float,
        required=True,
        help="Maximum learning rate for adam(default) optimizer. Initial learning \
            rate will be calculated according to warmup_ratio parameter \
            if use scheduler flag is 1. If the flag is 0, then this will \
            be initial learning rate"
    )
    parser.add_argument(
        "--use_scheduler",
        type=int,
        default=0,
        help="Flag to use scheduler or not. If 1 then use scheduler"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Ratio of total steps for warmup. If value is 0.2 and total steps is 100, then first 0.2*100 = 20 will be used for warmup"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="path to output directory. if directory doesn't exist then code will create the directory"
    )
    return parser


if __name__ == "__main__":
    parser = create_arguments()
    args = parser.parse_args()
    print(vars(args))
