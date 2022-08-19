import torch
import matplotlib.pyplot as plt
from scipy.special import rel_entr

class AttentionAnalyzer():
    def __init__(self, model=None):
        self.model = model
    
    @staticmethod
    def get_layer_attns(model, sentence, layer=1, avg_heads=True, avg_queries=True, only_CLS=False):
        outputs = model.predict([sentence], output_attentions=True, return_dict=True)
        attentions = outputs['attentions']
        print('a', attentions.size())
        att = attentions[layer-1].squeeze()
        print('b', att.size())
        if only_CLS:
            att = att[:,0,:]
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
    
    def attn_kl_div_all(self, o_sens, a_sens, layer=1):
        kls = []
        lengths = []
        for o, a in zip(o_sens, a_sens):
            kl, l = self._attn_kl_div(o, a, layer=layer)
            kls.append(kl)
            lengths.append(l)
        return kls, lengths

    def _attn_kl_div(self, sent_original, sent_attacked, layer=1):
        '''
        Calculate KL divergence between original and attacked attention distribution
        '''
        # get tokens
        tkns_original = self.model.tokenizer.tokenize(sent_original, add_special_tokens=True)
        tkns_attacked = self.model.tokenizer.tokenize(sent_attacked, add_special_tokens=True)
        seq_length = len(tkns_original)

        # Extract attention weights
        attns_original = self.get_layer_attns(self.model, sent_original, layer=layer, avg_heads=False, avg_queries=False, only_CLS=True).tolist()
        attns_attacked = self.get_layer_attns(self.model, sent_attacked, layer=layer, avg_heads=False, avg_queries=False, only_CLS=True).tolist()

        # match length of attn distributions
        attns_original, attns_attacked = self._match_length(attns_original, attns_attacked)

        # Calculate KL div
        kl_div = sum(rel_entr(attns_original, attns_attacked))

        # return KL div and length
        return kl_div, seq_length
    
    @staticmethod
    def _match_length(lst1, lst2):
        diff = abs(len(lst1) - len(lst2))
        if len(lst1) > len(lst2):
            tgt = lst1 
        else:
            tgt = lst2
        for _ in range(diff):
            tgt.remove(min(tgt))
        return lst1, lst2






