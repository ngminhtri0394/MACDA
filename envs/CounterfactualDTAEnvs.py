import torch
import torch.nn.functional as F
import mol_utils
from rdkit.Chem import QED
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
                 cof,
                 similarity_measure="neural_encoding",
                 **kwargs
                 ):
        super(CounterfactualDTAEnvs, self).__init__(**kwargs)

        Hyperparams = Args()
        self.device = device
        self.fp_length = Hyperparams.fingerprint_length
        self.fp_radius = Hyperparams.fingerprint_radius
        self.discount_factor = discount_factor
        self.model_to_explain = model_to_explain
        self.target_aff = torch.FloatTensor([[target_aff]])
        self.orig_pred = original_prediction
        self.distance = lambda x,y: F.l1_loss(x,y).detach()
        self.cof = cof
        self.base_loss = self.distance(self.orig_pred.cpu(), self.target_aff.cpu()).item()
        self.original_target = original_target
        self.original_molecule = original_molecule


        if similarity_measure == "neural_encoding":
            self.similarity = lambda x, y: cosine_similarity(x, y)
            [seq_cat(t) for t in self.original_target]

            _, self.drug_original_encoding, self.prot_original_encoding = self.model_to_explain(original_molecule.to(device), seq_cat(self.original_target).to(device))

        else:
            raise NotImplemented
    def reward(self, agent):
        molecule = Chem.MolFromSmiles(self._state[0])
        target = self._state[1]
        if molecule is None or len(molecule.GetBonds()) == 0:
            return 0.0, 0.0, 0.0
        new_mol_QED = QED.qed(molecule)
        molecule = mol_utils.mol_to_dta_pyg(molecule)
        # predict from pyg molecule and target
        pred, new_mol_encoding, new_prot_encoding = self.model_to_explain(molecule.to(self.device), seq_cat(target).to(self.device))
        pred_drug,_,_ = self.model_to_explain(molecule.to(self.device), seq_cat(self.original_target).to(self.device))
        pred_prot,_,_ = self.model_to_explain(self.original_molecule.to(self.device), seq_cat(target).to(self.device))

        #calculate sim between encoding of pyg molecule
        drug_sim = self.similarity(new_mol_encoding, self.drug_original_encoding)
        prot_sim = self.similarity(new_prot_encoding, self.prot_original_encoding)

        loss = self.distance(pred.cpu(), self.orig_pred.cpu()).item()
        drug_only_loss = self.distance(pred_drug.cpu(), self.orig_pred.cpu()).item()
        prot_only_loss = self.distance(pred_prot.cpu(), self.orig_pred.cpu()).item()

        gain = torch.sign(
            self.orig_pred - pred
        ).item()

        joint_loss = (loss * gain - drug_only_loss - prot_only_loss)

        reward = joint_loss * self.cof[0] + drug_sim * self.cof[1] + prot_sim * self.cof[2]

        del molecule, pred, new_mol_encoding
        return reward * self.discount_factor \
            ** (self.max_steps - self.num_steps_taken), joint_loss, gain, drug_sim, prot_sim, new_mol_QED

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
