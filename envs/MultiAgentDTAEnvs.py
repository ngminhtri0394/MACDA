from envs.DTAEnvs import DTAEnvs
import numpy as np
import mol_utils
class MultiAgentDTAEnvs:
    def __init__(self, envs:DTAEnvs,
                 observation_callback,
                 reward_callback,
                 done_callback,
                 reset_callback,
                 Hyperparams,
                 protein_seqlen):
        self._envs = envs
        self.observation_callback = observation_callback
        self.reward_callback = reward_callback
        self.done_callback = done_callback
        self.reset_callback = reset_callback
        self.agents = envs.agents
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            if agent.type == 'protein':
                self.action_space.append(len(envs.get_valid_actions_prot()))
                self.observation_space.append([1000])
            if agent.type == 'drug':
                self.action_space.append(len(envs.get_valid_actions_drug()))
                self.observation_space.append([Hyperparams.fingerprint_length])

    def step(self, actions):
        self.agents = self._envs.agents
        obs_n = []
        reward_n = []
        done_n = []
        self._envs.step(actions)

        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))
        return obs_n, reward_n, done_n

    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent)

    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent)

    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent)

    def reset(self):
        self.reset_callback()
        self.agents = self._envs.agents
        obs_n = []
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n




