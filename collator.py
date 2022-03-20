import torch


class Collator:
    """
        Collator Class.
        We pad the input ids with pad token id
        attention mask with 0 and tags with
        -100, as this index will be ignored by 
        nn.crossentropy function
        For mask, the code pads with value 0.
    """
    def __init__(self, tokenizer, MAX_LEN, ignore_index=-100):
        self.tokenizer = tokenizer
        self.MAX_LEN = MAX_LEN
        self.ignore_index = ignore_index

    def pad_inputs(self, inp, padlen):
        out = {}
        if self.tokenizer.padding_side == "right":
            # pad to right
            out["input_ids"] = inp["input_ids"] + [self.tokenizer.pad_token_id]*padlen
            out["attention_mask"] = inp["attention_mask"] + [0]*padlen
            out["tags"] = inp["tags"] + [self.ignore_index]*padlen
        elif self.tokenizer.padding_side == "left":
            # pad to left
            out["input_ids"] = [self.tokenizer.pad_token_id]*padlen + inp["input_ids"]
            uut["attention_mask"] = [0]*padlen + inp["attention_mask"]
            out["tags"] = [self.ignore_index]*padlen + inp["tags"]

        return out

    # custom collate function
    def collate_fn(self, inputs):
        # inputs will be list where each element is output of
        # dataset.__getitem__[index] which in our case
        # is dict with keys as id, input_ids, mask and tags and schema_tags.
        # see dataset.py for more details

        # for test data (which will go through same collator function)
        # is dict with keys only as id and input_ids. To make the code
        # work I will tags and schema_tags (which will be dummy in case of test data)
        for inp in inputs:
            length = len(inp["id"])
            if "tags" not in inp:
                inp["tags"] = ["dummy"]*length
            if "schema_tags" not in inp:
                inp["schema_tags"] = [-1]*length

        # first calculate the max len in the batch
        maxlen = -1
        for inp in inputs:
            maxlen = max(maxlen, len(inp["id"]))

        ids = []
        input_ids = []
        attention_mask = []
        tags = []

        if maxlen > self.MAX_LEN:
            # the current batch has examples greater than 512
            # if example length is less than 512, we will pad it to 512
            # if example length is greater than 512, we will chunk to make
            # multiple examples of length 512
            for inp in inputs:
                currlen = len(inp["tags"])
                if currlen > self.MAX_LEN:
                    x = range(0, currlen, self.MAX_LEN)
                    # the last one will be less than 512.
                    for idx in x[:-1]:
                        ids.append(inp["id"])
                        input_ids.append(inp["input_ids"][idx: idx + self.MAX_LEN])
                        attention_mask.append(inp["attention_mask"][idx: idx + self.MAX_LEN])
                        tags.append(inp["tags"][idx: idx + self.MAX_LEN])

                    # now pad the last one to 512
                    out = self.pad_inputs(
                        inp={
                            "input_ids": inp["input_ids"][x[-1]:],
                            "attention_mask": inp["attention_mask"][x[-1]:],
                            "tags": inp["tags"][x[-1]:]
                        },
                        padlen=self.MAX_LEN - len(inp["tags"][x[-1]:])
                    )
                    ids.append(inp["id"])
                    input_ids.append(out["input_ids"])
                    attention_mask.append(out["attention_mask"])
                    tags.append(out["tags"])
                else:
                    # pad it to 512
                    padlen = self.MAX_LEN - currlen
                    out = self.pad_inputs(inp, padlen)
                    ids.append(inp["id"])
                    input_ids.append(out["input_ids"])
                    attention_mask.append(out["attention_mask"])
                    tags.append(out["tags"])
        else:
            # pad all the samples to current max
            for inp in inputs:
                currlen = len(inp["tags"])
                padlen = maxlen - currlen
                out = self.pad_inputs(inp, padlen)
                ids.append(inp["id"])
                input_ids.append(out["input_ids"])
                attention_mask.append(out["attention_mask"])
                tags.append(out["tags"])

        return {
            "ids": ids,
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "tags": torch.tensor(tags)
        }
