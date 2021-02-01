import torch
import torch.nn.functional as F
import mol_utils

from config.explainer import Args
from rdkit import Chem, DataStructs
from envs import DTAEnvs
from mol_utils import tanimoto_similarity, rescaled_cosine_similarity, cosine_similarity
from mol_utils.preprocess import seq_cat
import numpy as np

class CounterfactualDTAEnvs(DTAEnvs.DTAEnvs):
    def __init__(self,
                 model_to_explain,
                 original_molecule,
                 original_target,
                 original_prediction,
                 discount_factor,
                 target_aff,
                 device,
                 similarity_set=None,
                 weight_sim=None,
                 similarity_measure="neural_encoding",
                 cof_sim=[0.6, 0.2, 0.2],
                 **kwargs
                 ):
        super(CounterfactualDTAEnvs, self).__init__(**kwargs)
        if weight_sim is None:
            weight_sim = 0.8
        Hyperparams = Args()
        self.device = device
        self.fp_length = Hyperparams.fingerprint_length
        self.fp_radius = Hyperparams.fingerprint_radius
        self.discount_factor = discount_factor
        self.model_to_explain = model_to_explain
        self.weight_sim = weight_sim
        self.target_aff = torch.FloatTensor([[target_aff]])
        self.orig_pred = original_prediction
        self.distance = lambda x,y: F.l1_loss(x,y).detach()
        self.base_loss = self.distance(self.orig_pred.cpu(), self.target_aff.cpu()).item()
        self.original_target = original_target
        self.cof_sim = cof_sim

        if similarity_measure == "rescaled_neural_encoding":
            self.similarity = lambda x, y: rescaled_cosine_similarity(x,
                                                                      y,
                                                                      similarity_set)
            self.drug_original_encoding = self.model_to_explain(original_molecule, self.original_target)[1]
        else:
            raise NotImplemented
    def reward(self, agent):
        molecule = Chem.MolFromSmiles(self._state[0])
        target = self._state[1]
        if molecule is None or len(molecule.GetBonds()) == 0:
            return 0.0, 0.0, 0.0

        molecule = mol_utils.mol_to_dta_pyg(molecule)
        # predict from pyg molecule and target
        pred, new_mol_encoding, new_prot_encoding = self.model_to_explain(molecule.to(self.device), seq_cat(target).to(self.device))
        #cal sim between encoding of pyg molecule
        drug_sim = self.similarity(new_mol_encoding, self.drug_original_encoding)
        prot_sim =  self.similarity(new_prot_encoding, self.prot_original_encoding)
        loss = self.distance(pred.cpu(), self.orig_pred.cpu()).item()
        gain = torch.sign(
            self.distance(pred.cpu(), self.target_aff.cpu()) - self.base_loss
        ).item()

        reward = gain * loss * self.cof_sim[0] + prot_sim * self.cof_sim[1] + drug_sim * self.cof_sim[2]

        del molecule, pred, new_mol_encoding
        return reward * self.discount_factor \
            ** (self.max_steps - self.num_steps_taken), loss, gain, drug_sim, prot_sim
        # return reward * self.discount_factor \
        #        ** (self.max_steps - self.num_steps_taken)

    def observation(self, agent):
        if agent.type == 'protein':
            prot_observations = seq_cat(self._state[1])
            return prot_observations
        elif agent.type == 'drug':
            drug_observations =mol_utils.numpy_morgan_fingerprint(
                            self._state[0],
                            self.Hyperparams.fingerprint_length,
                            self.Hyperparams.fingerprint_radius
                    )
            return drug_observations
        else:
            raise NotImplemented('Unrecognized agent.')

    def done(self, agent):
        return ((self._counter >= self.max_steps) or self._goal_reached())

    def reset(self):
        """Resets the MDP to its initial state."""
        self._state = [self.init_mol, self.init_prot]
        if self.record_path:
            self._path = [self._state]
        self._valid_actions_drug = self.get_valid_actions_drug(force_rebuild=True)
        self._valid_actions_prot = self.get_valid_actions_prot(force_rebuild=True)

        self._counter = 0
