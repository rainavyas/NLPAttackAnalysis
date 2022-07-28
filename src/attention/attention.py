import torch

class AttentionAnalyzer():
    def __init__(self, model=None):
        self.model = model
    
    @staticmethod
    def get_layer_attns(model, sentence, layer=1, avg_heads=True, avg_queries=True):
        outputs = model.predict([sentence], output_attentions=True, return_dict=True)
        attentions = outputs['attentions']
        att = attentions[layer-1].squeeze()
        if avg_heads:
            att = torch.mean(att, dim=0, keepdim=True)
        if avg_queries:
            att = torch.mean(att, dim=1, keepdim=True)
        return att.squeeze()


    
    @staticmethod
    def plot_attn_histogram(tkns_original, tkns_attacked, attns_original, attns_attacked, out_path_root, highlight_pos=None):
        '''
        Return
        '''
        pass

    def visualize_attack(self, sent_original, sent_attacked, out_path_root, layer=1, avg_heads=True, avg_queries=True):
        '''
        Generate histogram
        '''
        # get tokens
        tkns_original = self.model.tokenizer.tokenize(sent_original)
        tkns_attacked = self.model.tokenizer.tokenize(sent_attacked)
        print('tkns', len(tkns_original))

        # identify positions of difference
        pass

        # Extract attention weights
        attns_original = self.get_layer_attns(self.model, sent_original, layer=layer).tolist()
        attns_attacked = self.get_layer_attns(self.model, sent_attacked, layer=layer).tolist()
        print('attns', len(attns_original))

        # Generate plot
        self.plot_attn_histogram(tkns_original, tkns_attacked, attns_original, attns_attacked, out_path_root)


