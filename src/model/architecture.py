import torch 
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Config
from torch.nn import CrossEntropyLoss


class MultiTaskCodeT5(nn.Module):
    def __init__(self, model_name="Salesforce/codet5-base", num_classes=4):

        super().__init__()
        self.base = T5ForConditionalGeneration.from_pretrained(model_name)
        self.config= self.base.config

        self.classifier = nn.Sequential(
            nn.Linear(self.config.d_module, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask, labels=None, class_labels=None):

        outputs = self.base(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels=labels,
            output_hidden_state=True,
            return_dict=True
        )

        encoder_hidden = outputs.encoder_last_hidden_state

        cls_token = encoder_hidden[:, 0, :]
        logits_class=self.classifier(cls_token)


        return{
            "loss_gen": outputs.loss,
            "logits_gen": outputs.logits,
            "logits_class": logits_class,
            "encoder_hidden": encoder_hidden
        }
    
    
    def generate(self, input_ids, attention_mask, **kwargs):
        return self.base.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
    
    def compute_loss(self, outputs, class_labels):
        gen_loss=outputs["loss_gen"]
        class_loss_fct=CrossEntropyLoss()
        class_loss= class_loss_fct(outputs["logits_class"], class_labels)
        return 0.7 * gen_loss + 0.3 * class_loss