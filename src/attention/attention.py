import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.special import rel_entr
from scipy.stats import entropy
from statistics import mean

from ..tools.nw import nw
from ..tools.tools import distance

class AttentionAnalyzer():
    def __init__(self, model=None):
        self.model = model
        self.s = nn.Softmax(dim=0)
    
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
    def get_layer_embs(model, sentence, layer=1):
        '''
        Returns embeddings input to specified layer (i.e. what attention acts over)
        '''
        outputs = model.predict([sentence], output_hidden_states=True, return_dict=True)
        hidden_embs = outputs['hidden_states'][layer-1].squeeze()
        return hidden_embs
        
    
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
    
    def out_entropy_all(self, sens):
        '''
        Entropy of output distribution over classes
        '''
        ents = []
        for i, s in enumerate(sens):
            print(f'{i}/{len(sens)}')
            ent = self._out_entropy(s)
            ents.append(ent)
        return ents
    
    def emb_change_all(self, o_sens, a_sens, layer=1, dist='l2'):
        '''
        Average change in embedding vector (for substituted positions)
        '''
        diffs = []
        for i, (o, a) in enumerate(zip(o_sens, a_sens)):
            print(f'{i}/{len(o_sens)}')
            diff = self._emb_change(o, a, layer=layer, dist=dist)
            diffs.append(diff)
        return diffs
    
    def entropy_all(self, o_sens, a_sens, layer=1, num_heads=12, align=False):
        '''
        Entropy of attn sequence at layer, averaged over heads
        Uses CLS token as query for each attn

        If you want entropies for just o_sens, don't pass a_sens to function
        '''
        ents_o = []
        ents_a = []
        l_os = []
        l_as = []
        for i, (o, a) in enumerate(zip(o_sens, a_sens)):
            print(f'{i}/{len(o_sens)}')
            ent_o = 0
            ent_a = 0
            for h in range(num_heads):
                enth_o, enth_a, lo, la = self._attn_entropy(o, a, layer=layer, head=h, align=align)
                ent_o += enth_o
                ent_a += enth_a
            ents_o.append(ent_o/num_heads)
            ents_a.append(ent_a/num_heads)
            l_os.append(lo)
            l_as.append(la)
        return ents_o, ents_a, l_os, l_as
    
    def attn_kl_div_all(self, o_sens, a_sens, layer=1, num_heads=12):
        '''
        For each orig-attack pair get KL div per layer, where KL avg over heads
        Uses CLS token as query for each attn
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
    
    def _out_entropy(self, sen):
        '''
        Entrpy of output class distribution
        '''
        logits = self.model.predict(sen).squeeze()
        probs = self.s(logits)
        entropy = entropy(probs.tolist())
        return entropy
    
    def _emb_change(self, sent_original, sent_attacked, layer=1, dist='l2'):
        '''
        Calculate average (over substitutions) change in embedding values
        '''
        # get tokens
        tkns_original = self.model.tokenizer.encode(sent_original, add_special_tokens=True)
        tkns_attacked = self.model.tokenizer.encode(sent_attacked, add_special_tokens=True)

        # get embeddings
        embs_original = self.get_layer_embs(self.model, sent_original, layer=layer)
        embs_attacked = self.get_layer_embs(self.model, sent_attacked, layer=layer)

        # token alignment
        orig_seq, att_seq = nw([str(t) for t in tkns_original], [str(t) for t in tkns_attacked])

        # Calculate average embedding change
        diffs = []
        count_o = 0
        count_a = 0
        for o, a in zip(orig_seq, att_seq):
            if o == a:
                count_o += 1
                count_a += 1
            else:
                if o != '-' and a != '-':
                    diffs.append(distance(embs_original[count_o].squeeze(), embs_attacked[count_a].squeeze(), typ=dist).item())
                    count_o += 1
                    count_a += 1
                elif o == '-':
                    count_a += 1
                elif a == '-':
                    count_o += 1
                else:
                    print("Alignment went wrong")
        if len(diffs) == 0:
            return 0
        return mean(diffs)


    
    def _attn_entropy(self, sent_original, sent_attacked, layer=1, head=0, align=False):
        '''
        Calculate entropy of attn distribution at specified layer and head
        Use CLS token as query for attn
        '''
        if align:
            attns_original, attns_attacked, _ = self._align(sent_original, sent_attacked, layer=layer, head=head)
        else:
            attns_original = self.get_layer_attns(self.model, sent_original, layer=layer, avg_heads=False, avg_queries=False, only_CLS=True).tolist()[head]
            attns_attacked = self.get_layer_attns(self.model, sent_attacked, layer=layer, avg_heads=False, avg_queries=False, only_CLS=True).tolist()[head]
        return entropy(attns_original), entropy(attns_attacked), len(attns_original), len(attns_attacked)

    def _attn_kl_div(self, sent_original, sent_attacked, layer=1, head=0):
        '''
        Calculate KL divergence between original and attacked attention distribution
        Use CLS token as query for attn
        '''
        attns_original, attns_attacked, _ = self._align(sent_original, sent_attacked, layer=layer, head=head)

        # Calculate KL div
        kl_div = sum(rel_entr(attns_original, attns_attacked))

        # return KL div and length
        return kl_div, len(attns_original)
    
    def _align(self, sent_original, sent_attacked, layer=1, head=0):
        '''
        Sentences to aligned attn sequences
        Use CLS Token for attn query
        '''
        # get tokens
        tkns_original = self.model.tokenizer.encode(sent_original, add_special_tokens=True)
        tkns_attacked = self.model.tokenizer.encode(sent_attacked, add_special_tokens=True)

        # Extract attention weights
        attns_original = self.get_layer_attns(self.model, sent_original, layer=layer, avg_heads=False, avg_queries=False, only_CLS=True).tolist()[head]
        attns_attacked = self.get_layer_attns(self.model, sent_attacked, layer=layer, avg_heads=False, avg_queries=False, only_CLS=True).tolist()[head]

        # match length of attn distributions
        return self._match_length(attns_original, attns_attacked, tkns_original, tkns_attacked)
    
    
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
        
        orig_seq, att_seq = nw([str(t) for t in tkns_orig], [str(t) for t in tkns_att])
        
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
                    












