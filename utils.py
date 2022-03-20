import os
import sys
import tqdm
import torch
import numpy as np


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class MovingAverage:
    def __init__(self, name, rd=4):
        self.name = name
        # avg value
        self.value = 0
        self.sum = 0
        self.count = 0
        self.rd = rd

    def update(self, x):
        self.sum += x
        self.count += 1

        # update self.value
        self.value = self.sum / self.count
        # round off the value
        self.value = round(self.value, self.rd)


def calculate_loss(loss_fn, logits, targets):
    """
    this is will be used by train and evaluate.
    so I thought to have one common function for both of them
    other than this has no special purpose
    """
    # the reason of why I am doing this is : https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    # Bascially cross entropy needs in (*, classes) and (*) to compute
    # I copied this idea from somewhere else (which is obvious)
    # we will receive (batch_size, seq_len, num_classes) and (batch_size, seq_len)
    num_classes = logits.shape[-1]
    # so we will compute the loss as (batch_size*seq_len, num_classes) and (batch_size*seq_len,)
    loss = loss_fn(logits.view(-1, num_classes), targets.view(-1))
    return loss


# function to return merged ground truth and prediction for batch
def merge_prediction(ids, logits):
    assert len(ids) == len(logits)
    # logits will be of shape (batch_size, seq_len, num_classes)
    # logits will have gradient.
    logits = logits.detach().cpu().numpy()
    # dictionary with id as the key
    pred = {}
    for index, chunk in enumerate(logits):
        # chunk will of shape (seq_len, num_classes)
        if ids[index] in pred:
            pred[ids[index]] = np.hstack((pred[ids[index]], chunk))
        else:
            pred[ids[index]] = chunk

    return pred


def merge_ground_truth(ids, tags):
    # tags will be of shape (batch_size, seq_len)
    assert len(ids) == len(tags)
    tags = tags.cpu().numpy()
    # dictionary with id as the key
    true = {}
    for index, chunk in enumerate(tags):
        # chunk will of shape (seq_len, )
        if ids[index] in true:
            true[ids[index]] = np.concatenate((true[ids[index]], chunk))
        else:
            true[ids[index]] = chunk

    return true


# function for running one pass in dataloder
# steps argument is used only for logging
def train(
    model,
    loss_fn,
    optimizer,
    scheduler,
    dataloder,
    device,
    gradient_accumulation_steps=1,
    steps=None,
    use_scheduler=True,
    verbose=True
):
    if steps is None:
        # this will give the number of steps
        steps = len(dataloder)

    train_loss = MovingAverage(name="train_loss")
    optimizer.zero_grad()
    model.train()

    for batch, inputs, in enumerate(dataloder):
        input_ids = inputs["input_ids"].long().to(device)
        attention_mask = inputs["attention_mask"].long().to(device)
        target = inputs["tags"].long().to(device)
        optimizer.zero_grad()
        # output from linear layer
        logits = model.forward(input_ids, attention_mask)
        loss = calculate_loss(loss_fn, logits, target)
        loss.backward()

        if (batch + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

            if use_scheduler:
                scheduler.step()

        # update the moving average
        train_loss.update(loss.item())

        # if verbose, then print to sys.stdout
        if verbose:
            out_log = "{}/{} {} = {}".format(batch + 1, steps, train_loss.name, train_loss.value)
            if batch + 1 != steps:
                sys.stdout.write(out_log + "\r")
            else:
                sys.stdout.write(out_log + "\n")

            sys.stdout.flush()

    # return the final loss value
    return train_loss.value


def evaluate(model, loss_fn, dataloder, device, steps=None, verbose=True):
    if steps is None:
        # this will give number of steps
        steps = len(dataloder)

    true, preds = {}, {}
    val_loss = MovingAverage(name="val_loss")

    model.eval()
    with torch.no_grad():
        for batch, inputs in enumerate(dataloder):
            ids = inputs["ids"]
            input_ids = inputs["input_ids"].long().to(device)
            attention_mask = inputs["attention_mask"].long().to(device)
            target = inputs["tags"].long().to(device)
            # output from linear layer
            logits = model.forward(input_ids, attention_mask)
            loss = calculate_loss(loss_fn, logits, target)

            # update the moving average
            val_loss.update(loss.item())

            # now save ground truth and prediction
            # if you want save more.
            preds = {**preds, **merge_prediction(ids, logits)}
            true = {**true, **merge_ground_truth(ids, target)}

            if verbose:
                out_log = "{}/{} {} = {}".format(batch + 1, steps, val_loss.name, val_loss.value)
                if batch + 1 != steps:
                    sys.stdout.write(out_log + "\r")
                else:
                    sys.stdout.write(out_log + "\n")

                sys.stdout.flush()

    # loss, prediction and ground truth
    return val_loss.value, preds, true


def predict(model, dataloader, steps, verbose=True):
    if steps is None:
        steps = len(dataloader)

    preds = {}

    model.eval()
    with torch.no_grad():
        for batch, inputs in enumerate(dataloader):
            ids = inputs["ids"]
            input_ids = inputs["input_ids"].long().to(device)
            attention_mask = inputs["attention_mask"].long().to(device)
            # output from linear layer
            logits = model.forward(input_ids, attention_mask)

            preds = {**preds, **merge_prediction(ids, logits)}

            if verbose:
                out_log = "{}/{}".format(batch + 1, steps)
                if batch + 1 != steps:
                    sys.stdout.write(out_log + "\r")
                else:
                    sys.stdout.write(out_log + "\n")

                sys.stdout.flush()

    return preds
