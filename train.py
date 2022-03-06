import os
import torch
import datetime
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import RobertaConfig, RobertaTokenizerFast, RobertaModel

import utils
import arguments
from models import FeedbackPriceModel
from dataset import FeedbackPriceDataset
from collator import Collator


def test_tokenization(tokenizer):
    # this function is written to test on small input string
    # how the tokenization and bies tags are generated
    # this is what I will do for every sample in the directory.
    
    # sample sentence and tags
    sentence = "Hello worldy people ok you know what! of coursey"
    tags = ["O", "L", "C", "C", "C", "C", "O", "M", "L"]

    print("Running the tokenize to see the output tokens")
    print("Tokens = {}", tokenizer.tokenize(sentence))

    # this is main function where I am converting the sentence to ids, and aligning the
    # tags to bies schema.
    tokenized_sentence, bies_tags = utils.tokenize(sentence, tags, tokenizer)
    print("Tokenized sentence = {}".format(tokenized_sentence))
    print("Bies tags = {}".format(bies_tags))


def main(args):
    # find the device and set the seed
    seed = 33
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    utils.set_seed(seed)

    # read the csv path
    df = pd.read_csv(args.train_csv_path)

    # take the unique labels
    # add them to BIES schema
    labels = []
    for label in df["discourse_type"].unique():
        labels.extend(["{}-{}".format(sch, label) for sch in "BIES"])

    labels.append("O")

    label2idx = {label: index for index, label in enumerate(labels)}
    idx2label = {index: label for index, label in enumerate(labels)}
    print("Number of labels = {}".format(len(labels)))
    print("Label to index = {}".format(label2idx))

    # read the directory
    print("Reading the files from {} directory".format(args.train_dir))
    texts = utils.return_texts(args.train_dir)
    num_documents = len(os.listdir(args.train_dir))

    print("Number of documents found in directory = {}".format(num_documents))
    print("Combining the read files with ground truth from csv")
    texts_mapping = utils.return_texts_mapping(texts, df)

    # initialize the tokenizer
    # add_prefix_space is false by default
    print("Initializing the {} tokenizer".format(args.model_name_or_path))
    tokenizer = RobertaTokenizerFast.from_pretrained(
        args.model_name_or_path,
        add_prefix_space=False
    )
    print("Special tokens")
    print(dict(zip(tokenizer.all_special_tokens, tokenizer.all_special_ids)))

    # run the test tokenization function
    test_tokenization(tokenizer)

    # convert the texts to ids and also align the tags with bies schema
    # remember that all the data is loaded here only
    inputs = utils.prepare_inputs(texts, texts_mapping, tokenizer)

    # load the pretrained model
    print("Initializing the config")
    roberta_config = RobertaConfig.from_pretrained(args.model_name_or_path)
    # set in roberta config to output hidden states
    roberta_config.output_hidden_states = True
    print("Initializing the {} as base model".format(args.model_name_or_path))
    # while using like this, the model is set in evaluation by default.
    roberta_model = RobertaModel.from_pretrained(args.model_name_or_path, config=roberta_config)
    print("Initializing the finetuing model using base model")
    model = FeedbackPriceModel(roberta_config, roberta_model, len(label2idx))
    model.to(device)
    print("Model Initialization completed")

    # splitting the data into train and validation
    ids = np.array(list(inputs.keys()))
    total_size = len(ids)

    # shuffle the ids
    np.random.shuffle(ids)

    train_ids = ids[:int(0.8*total_size)]
    val_ids = ids[int(0.8*total_size):]

    train_size = len(train_ids)
    val_size = len(val_ids)

    print("Total size = {}".format(total_size))
    print("Train size = {}".format(train_size))
    print("Validation size = {}".format(val_size))

    # initialize the ids
    train_dataset = FeedbackPriceDataset(train_ids, inputs, label2idx)
    val_dataset = FeedbackPriceDataset(val_ids, inputs, label2idx)

    # collator and dataloader.
    collator = Collator(tokenizer, args.max_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator.collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator.collate_fn
    )

    # calcuation of steps and warmup steps
    train_steps_per_epoch = int(np.ceil(train_size / args.batch_size))
    val_steps_per_epoch = int(np.ceil(val_size / args.batch_size))
    total_steps = train_steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)

    print("Train Steps per epoch = {}".format(train_steps_per_epoch))
    print("Val Steps per epoch = {}".format(val_steps_per_epoch))
    print("Total steps = {}".format(total_steps))
    print("Warmpup steps = {}".format(warmup_steps))
    # calculate initial learning rate.
    args.inital_learning_rate = args.max_learning_rate / warmup_steps
    print("Initial Learning rate = {}".format(args.inital_learning_rate))

    # optimizer
    print("Initializing the optimizer")
    optimizer = optim.Adam(model.parameters(), lr=args.inital_learning_rate, betas=(0.9, 0.98), eps=1e-9)
    optimizer.zero_grad()

    # scheduler
    # as soon as you initialize, scheduler.step() will be called in __init__ function.
    # as this is LambdaLR, it will multiple base_lr with the lambda function you have given
    # to compute lambda function, what it will do is lambda_fn(last_epoch + 1)
    # as last_epoch is -1, it means our lambda_fn will be evaluated at 0 at init stage
    """
    This is how psudecode looks for lambda lr step method
    # inputs is base_lr, lambda_fn, last_epoch
    def step():
        last_epoch += 1
        multiply = lambda_fn(last_epoch)
        lr = base_lr*multiply
        return lr
    """
    # so next scheduler.step() will increase the last_epoch to 1 and call lambda_fn(1) and multiply with base_lr
    # last_epoch can be used to continue the training from a checkpoint.
    # source code : https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#LambdaLR
    # scheduler will have method called scheduler.get_lr which will give the last calculated lr
    # this can be used to log the learning rate.
    print("Initializing the scheduler")
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((warmup_steps**2)/(step + 1), (step + 1)),
        last_epoch=-1,
        verbose=False
    )

    # loss function
    # -100 is the default value
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # training
    for epoch in range(args.epochs):
        print("*"*60)
        print("{}/{}".format(epoch + 1, args.epochs))

        t1 = datetime.datetime.now()
        # train
        train_loss = utils.train(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloder=train_loader,
            device=device,
            steps=train_steps_per_epoch,
            verbose=True
        )

        # validation
        val_loss, val_preds, val_true = utils.evalute(
            model=model,
            loss_fn=loss_fn,
            dataloder=val_loader,
            device=device,
            steps=val_steps_per_epoch,
            verbose=True
        )
        t2 = datetime.datetime.now()
        timetaken = round((t2 - t1).total_seconds())
        print("time: {}, loss: {}, val loss: {}".format(timetaken, train_loss, val_loss))


if __name__ == "__main__":
    parser = arguments.create_arguments()
    args = parser.parse_args()
    print("Parsed Arguments = {}".format(args))
    main(args)
