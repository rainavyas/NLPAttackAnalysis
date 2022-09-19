import numpy as np


class Retention():
    @staticmethod
    def retention_curve_frac_positive(vals, labels, ascending=True):
        '''
            vals: List of values threshold retention fraction upon
            labels: List of Binary positive (1) or negative class (1)

            - For retention fraction, r, select the smallest/largest samples by vals
            - For the retain samples, calculate the fraction, f(r), of the positive sample recalled from labels
                i.e, if labels has 10 positive samples, and at retention r, we have 3 samples, recall fraction is 0.3
            
            - Repeat for r from 0 to 1, and return f(r)
        '''
        rets = [(i+1)/len(vals) for i,_ in enumerate(vals)]
        ordering = np.argsort(np.asarray(vals))
        ordered_labels = np.asarray(labels)[ordering]

        counts = []
        curr = 0
        for l in ordered_labels:
            if l==1:
                curr+=1
            counts.append(curr)
        recalls = [c/counts[-1] for c in counts]

        return recalls, rets
        

    
    pass
