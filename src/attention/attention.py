
class AttentionAnalyzer():
    def __init__(self, model=None):
        self.model = model
    
    @staticmethod
    def get_layer_attns(self, model, sentence, layer=1, avg_heads=True, avg_queries=True):
        outputs = model.predict([sentence], output_attentions=True, return_dict=True)
        import pdb; pdb.set_trace()
    
    @staticmethod
    def plot_attn_histogram(attns_original, attns_attacked, out_file, highlight_pos=None):
        '''
        Return
        '''
        pass

    def visualize_attack(self, sent_original, sent_attack, out_file, layer=1, avg_heads=True, avg_queries=True):
        '''
        Generate histogram
        '''
        # get input ids

        # identify positions of difference
        pass

        # Extract attention weights
        self.get_layer_attns(self.model, sent_original, layer=layer)

        # Generate plot


