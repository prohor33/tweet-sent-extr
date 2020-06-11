import transformers
import torch.nn as nn
import torch


class TweetModel(transformers.BertPreTrainedModel):
    def __init__(self, bert_conf, global_conf):
        super(TweetModel, self).__init__(bert_conf)
        self.roberta = transformers.RobertaModel.from_pretrained(global_conf.bert_path, config=bert_conf)
        self.drop_out = nn.Dropout(0.17)
        self.l0 = nn.Linear(bert_conf.hidden_size * 2, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)

    def forward(self, ids, mask, token_type_ids):
        _, _, out = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        out = torch.cat((out[-1], out[-2]), dim=-1)
        out = self.drop_out(out)
        logits = self.l0(out)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits