from torch import nn
from transformers import *
import torch
from torch.nn import CrossEntropyLoss


# In[3]:
class NLPClassification(torch.nn.Module):
    def __init__(self, **args):
        super(NLPClassification, self).__init__()
        self.args = args
        self.base_model = BertModel.from_pretrained(self.args.base_model)
        self.classifier = torch.nn.Linear(self.base_model.config.hidden_size, self.args.num_labels)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                ):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        h = self.dropout(outputs.pooler_output)
        logits = self.classifier(h)
        if labels is None:
            return logits
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.args.num_labels), labels.view(-1))
        return loss
