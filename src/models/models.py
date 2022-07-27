from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn

class SequenceClassifier(nn.Module):
    def __init__(self, model_name='bert-base-cased', num_labels=2):
        super().__init__()
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)[0]
    
    def predict(self, sentences, output_attentions=False, output_hidden_states=False, return_dict=False):
        inputs = self.tokenizer(sentences, padding=True, max_length=512, truncation=True, return_tensors="pt")
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        return(self.model(input_ids, attention_mask, output_attentions=output_attentions,
                 output_hidden_states=output_hidden_states, return_dict=return_dict))



        
