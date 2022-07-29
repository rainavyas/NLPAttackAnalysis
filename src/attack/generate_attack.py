'''
    Refer here for text attack recipes:
         https://textattack.readthedocs.io/en/latest/1start/attacks4Components.html
'''

import textattack
from model_wrapper import PyTorchModelWrapper
from .redefined_textattack_models import DeepWordBugGao2018


class Attacker():
    def __init__(self, model, attack_recipe='pwws', constraint=None):
        model_wrapper = PyTorchModelWrapper(model, model.tokenizer)
        if attack_recipe == 'pwws':
            # no constraint enforced as imperceptibility defined by synonym substitution
            self.attack = textattack.attack_recipes.pwws_ren_2019.PWWSRen2019.build(model_wrapper)
        elif attack_recipe == 'bae':
            if not constraint:
                self.attack = DeepWordBugGao2018.build(model_wrapper)
            else:
                self.attack = DeepWordBugGao2018.build(model_wrapper, levenstein_dist=constraint)

    def attack_sentence(self, sentence, label):
        attack_result = self.attack.attack(sentence, label)
        updated_sentence = attack_result.perturbed_text()
        return updated_sentence

