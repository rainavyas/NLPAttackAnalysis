import torch
import matplotlib.pyplot as plt
from scipy.special import rel_entr
from ..tools.nw import nw

class AttentionAnalyzer():
    def __init__(self, model=None):
        self.model = model
    
    @staticmethod
    def get_layer_attns(model, sentence, layer=1, avg_heads=True, avg_queries=True, only_CLS=False):
        outputs = model.predict([sentence], output_attentions=True, return_dict=True)
        attentions = outputs['attentions']
        att = attentions[layer-1].squeeze()
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
    
    def attn_kl_div_all(self, o_sens, a_sens, layer=1, num_heads=12):
        '''
        For each orig-attack pair get KL div per layer, where KL avg over heads
        '''
        kls = []
        lengths = []
        for i, (o, a) in enumerate(zip(o_sens, a_sens)):
            print(f'{i}/{len(o_sens)}')
            kl = 0
            for h in range(num_heads):
                klh, l = self._attn_kl_div(o, a, layer=layer, head=h)
                kl += klh
            kl = kl/num_heads
            kls.append(kl)
            lengths.append(l)
        return kls, lengths

    def _attn_kl_div(self, sent_original, sent_attacked, layer=1, head=0):
        '''
        Calculate KL divergence between original and attacked attention distribution
        '''
        # get tokens
        tkns_original = self.model.tokenizer.tokenize(sent_original, add_special_tokens=True)
        tkns_attacked = self.model.tokenizer.tokenize(sent_attacked, add_special_tokens=True)
        seq_length = len(tkns_original)

        # Extract attention weights
        attns_original = self.get_layer_attns(self.model, sent_original, layer=layer, avg_heads=False, avg_queries=False, only_CLS=True).tolist()[head]
        attns_attacked = self.get_layer_attns(self.model, sent_attacked, layer=layer, avg_heads=False, avg_queries=False, only_CLS=True).tolist()[head]

        # match length of attn distributions
        attns_original, attns_attacked, _ = self._match_length(attns_original, attns_attacked, tkns_original, tkns_attacked)

        # Calculate KL div
        kl_div = sum(rel_entr(attns_original, attns_attacked))

        # return KL div and length
        return kl_div, seq_length
    
    # @staticmethod
    # def _match_length(lst1, lst2):
    #     diff = abs(len(lst1) - len(lst2))
    #     if len(lst1) > len(lst2):
    #         tgt = lst1 
    #     else:
    #         tgt = lst2
    #     for _ in range(diff):
    #         tgt.remove(min(tgt))
    #     return lst1, lst2
    
    @staticmethod
    def _match_length(attns_orig, attns_att, tkns_orig, tkns_att):
        '''
            Align using the needleman_wunsch alignment algorithm
            Return positions of difference
            Distribute attention weights
                 - where there is extra token in one sequence merge attn prob into previous token
        
        e.g. 
            1) G  -  A  T  T  A  C  A        --->    G  AT T  A  C  A
            2) G  C  A  -  T  G  C  U        --->    GC A  T  G  C  U
        '''
        orig_seq, att_seq = nw(tkns_orig, tkns_att)
        print(orig_seq, att_seq)
        
        new_attns_orig = []
        new_attns_att = []
        inds = []

        count_o = 0
        count_a = 0

        for o, a in zip(orig_seq, att_seq):
            if o == a:
                new_attns_orig.append(attns_orig[count_o])
                new_attns_att.append(attns_att[count_a])
                count_o += 1
                count_a += 1
            else:
                if o != '-' and a != '-':
                    new_attns_orig.append(attns_orig[count_o])
                    new_attns_att.append(attns_att[count_a])
                    count_o += 1
                    count_a += 1
                    inds.append(len(new_attns_orig)-1)
                elif o == '-':
                    new_attns_att[-1] += attns_att[count_a]
                    count_a += 1
                    inds.append(len(new_attns_orig)-1)
                elif a == '-':
                    new_attns_orig[-1] += attns_orig[count_o]
                    count_o += 1
                    inds.append(len(new_attns_orig)-1)
                else:
                    print("Alignment went wrong")
        return new_attns_orig, new_attns_att, inds
                    












