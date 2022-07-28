import torch
import matplotlib.pyplot as plt

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
        out_file = f'{out_path_root}_original.png'
        plt.bar(tkns_original, attns_original)
        plt.xticks(rotation = 90, fontsize='xx-small')
        plt.savefig(out_file, bbox_inches='tight')
        plt.clf()
    
        out_file = f'{out_path_root}_attacked.png'
        plt.bar(tkns_attacked, attns_attacked)
        plt.xticks(rotation = 90, fontsize='xx-small')
        plt.savefig(out_file, bbox_inches='tight')
        plt.clf()

    def visualize_attack(self, sent_original, sent_attacked, out_path_root, layer=1, avg_heads=True, avg_queries=True):
        '''
        Generate histogram
        '''
        # get tokens
        # tkns_original = ['START'] + self.model.tokenizer.tokenize(sent_original) + ['END']
        # tkns_attacked = ['START'] + self.model.tokenizer.tokenize(sent_attacked) + ['END']
        tkns_original = self.model.tokenizer.tokenize(sent_original, add_special_tokens=True)
        tkns_attacked = self.model.tokenizer.tokenize(sent_attacked, add_special_tokens=True)

        # identify positions of difference
        pass

        # Extract attention weights
        attns_original = self.get_layer_attns(self.model, sent_original, layer=layer).tolist()
        attns_attacked = self.get_layer_attns(self.model, sent_attacked, layer=layer).tolist()
        
        # Generate plot
        assert len(tkns_original) == len(attns_original), "Mismatch in num tokens and attn weights"
        self.plot_attn_histogram(tkns_original, tkns_attacked, attns_original, attns_attacked, out_path_root)


