import torch
import torch.nn as nn


class FeedbackPriceModel(nn.Module):
    def __init__(self, base_config, base_model, num_classes):
        super(FeedbackPriceModel, self).__init__()
        self.base_config = base_config
        self.num_classes = num_classes

        self.base_model = base_model
        self.linear = nn.Linear(in_features=self.base_config.hidden_size, out_features=num_classes)

    def forward(self, input_ids, attention_mask):
        base_model_out = self.base_model(input_ids, attention_mask)
        # see : https://huggingface.co/docs/transformers/model_doc/roberta#transformers.RobertaModel
        # we will get the tuple of embedding output + all layer output
        # we will take last layer as of now.
        h = base_model_out.hidden_states[-1]
        out = self.linear(h)

        return out
