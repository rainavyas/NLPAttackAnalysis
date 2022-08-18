'''
    Refer here for text attack recipes:
         https://textattack.readthedocs.io/en/latest/1start/attacks4Components.html
'''

import torch
import textattack
from .model_wrapper import PyTorchModelWrapper
from .redefined_textattack_models import DeepWordBugGao2018, TextFoolerJin2019
import numpy as np



class Attacker():
    def __init__(self, model, attack_recipe='pwws', use_constraint=None, lev_dist_constraint=None):
        self.model = model
        model_wrapper = PyTorchModelWrapper(model, model.tokenizer)
        if attack_recipe == 'pwws':
            # no constraint enforced as imperceptibility defined by synonym substitution
            self.attack = textattack.attack_recipes.pwws_ren_2019.PWWSRen2019.build(model_wrapper)
        elif attack_recipe == 'deepwordbug':
            if not lev_dist_constraint:
                self.attack = DeepWordBugGao2018.build(model_wrapper)
            else:
                self.attack = DeepWordBugGao2018.build(model_wrapper, levenstein_dist=lev_dist_constraint)
        elif attack_recipe == 'textfooler':
            if not use_constraint:
                self.attack = TextFoolerJin2019.build(model_wrapper)
            else:
                self.attack = TextFoolerJin2019.build(model_wrapper, use_constraint=use_constraint)

    def attack_sentence(self, sentence, label):
        attack_result = self.attack.attack(sentence, label)
        updated_sentence = attack_result.perturbed_text()

        with torch.no_grad():
            logits = self.model.predict([sentence])[0].squeeze()
            orig_pred_class = torch.argmax(logits)
            logits = self.model.predict([updated_sentence])[0].squeeze()
            attacked_pred_class = torch.argmax(logits)
        return updated_sentence, orig_pred_class, attacked_pred_class
    
    @classmethod
    def fooling_rate(cls, o_preds, a_preds, labels):
        '''
        Fraction of correctly classified points that are mis-classified after attack
        '''
        success, unsuccess = cls.get_success_and_unsuccess_attacks([0]*len(o_preds), [0]*len(o_preds), o_preds, a_preds, labels)
        fool_rate = len(success['labels'])/(len(success['labels']) + len(unsuccess['labels']))
        return fool_rate
    
    @staticmethod
    def get_success_and_unsuccess_attacks(o_sens, a_sens, o_preds, a_preds, labels):
        '''
        Filters out mis-classified samples
        Splits data into successful and unsuccessful attacks
        '''
        o_preds = np.array(o_preds)
        a_preds = np.array(a_preds)
        labels = np.array(labels)
        s_inds = []
        u_inds = []
        
        for ind, (o,a,l) in enumerate(zip(o_preds, a_preds, labels)):
            if o == l:
                if o != a:
                    s_inds.append(ind)
                else:
                    u_inds.append(ind)

        s_o_sens = [o_sens[i] for i in s_inds]
        s_a_sens = [a_sens[i] for i in s_inds]
        u_o_sens = [o_sens[i] for i in u_inds]
        u_a_sens = [a_sens[i] for i in u_inds]
        success = {'o_sens':s_o_sens, 'a_sens':s_a_sens, 'o_preds':o_preds[s_inds], 'a_preds':a_preds[s_inds], 'labels':labels[s_inds]}
        unsuccess = {'o_sens':u_o_sens, 'a_sens':u_a_sens, 'o_preds':o_preds[u_inds], 'a_preds':a_preds[u_inds], 'labels':labels[u_inds]}
        return success, unsuccess


