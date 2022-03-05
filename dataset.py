from torch.utils.data import Dataset


class FeedbackPriceDataset(Dataset):
    def __init__(self, ids, inputs, label2idx):
        self.ids = ids
        self.inputs = inputs
        self.label2idx = label2idx

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        inp = self.inputs[self.ids[index]]
        return {
            "id": self.ids[index],
            "input_ids": inp["tokens"]["input_ids"],
            "attention_mask": inp["tokens"]["attention_mask"],
            "tags": [self.label2idx[tag] for tag in inp["tags"]],
            "schema_tags": inp["tags"]
        }
