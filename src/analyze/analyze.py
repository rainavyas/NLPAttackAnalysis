import torch


from ..tools.tools import batch_generator

class Analyzer():
    '''
    Suite of methods for analysis
    '''
    def __init__(self, model, device=torch.device('cpu')) -> None:
        self.model = model
        self.device = device
    
    
    def train_gaussian(self, sentences, bs=8):
        '''
        Learn Gaussian distribution (mean and variance) 
        on sentence embedding (layer 12 CLS token)
        '''
        cls = self._sens_to_clss(sentences, bs=bs)
        self.mean = torch.mean(cls, dim=0)
        self.cov = torch.cov(cls)
        import pdb; pdb.set_trace()
    
    def eval_gaussian(self, sentences, bs=8):
        '''
        Return likelihood of all samples passed
        '''
        cls = self._sens_to_clss(sentences, bs=bs)
        dist = torch.distributions.multivariate_normal.MultivariateNormal(self.mean, covariance_matrix=self.cov)
        log_probs = dist.log_prob(cls) # need to check this works batchwise
        return log_probs.detach().cpu().tolist()

    def _sens_to_clss(self, sentences, bs=8):
        '''
        Return torch tensor of cls embeddings (B x 768)
        '''
        all_cls = []
        for curr_sens in batch_generator(sentences, bs):
            with torch.no_grad():
                out = self.model.predict(curr_sens, output_hidden_states=True, return_dict=True, device=self.device)
                embs = out['hidden_states'][-1]
                all_cls.append(embs[:,0,:].squeeze(dim=1))
        cls = torch.cat(all_cls, dim=0).cpu()
        return cls


