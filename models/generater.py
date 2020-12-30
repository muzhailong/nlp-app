from torch import nn
from transformers import *
from torch.nn import CrossEntropyLoss


class ConditionalGenerationModel(nn.Module):
    def __init__(self, **args):
        super(ConditionalGenerationModel, self).__init__()
        self.args = args
        if self.args.base_model.endswith(".json"):
            # 使用config
            model_config = GPT2Config.from_json_file(args.base_model)
            self.base_model = GPT2LMHeadModel(config=model_config)
        else:
            # 载预训练模型
            self.base_model = GPT2LMHeadModel.from_pretrained(args.base_model)
        self.base_model.resize_token_embeddings(len(args.tokenizer))
        self.config = self.base_model.config

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        r"""
        labels 其中为title部分为token的id 其他部分为0 计算loss时可以使用ignore_index 忽略掉
        """
        output = self.base_model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        lm_logits = output.logits
        if labels is None:
            return lm_logits

        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss(reduction='sum', ignore_index=self.args.tokenizer.pad_token_id)  # [PAD]=0
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        num = shift_labels.ne(0).long().sum().item()
        loss = loss / num
        return loss
