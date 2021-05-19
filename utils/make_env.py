"""
Code for creating a multiagent environment with one of the scenarios listed
in ./scenarios/.
Can be called by using, for example:
    env = make_env('simple_speaker_listener')
After producing the env object, can be used similarly to an OpenAI gym
environment.

A policy using this environment must output actions in the form of a list
for all agents. Each element of the list should be a numpy array,
of size (env.world.dim_p + env.world.dim_c, 1). Physical actions precede
communication actions in this array. See environment.py for more details.
"""
from envs.CounterfactualDTAEnvs import CounterfactualDTAEnvs
from envs.MultiAgentDTAEnvs import MultiAgentDTAEnvs


def make_env(original_drug_smile, original_target, Hyperparams, atoms_, model_to_explain, original_drug,
             original_target_aff, pred_aff, device, cof):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    world = CounterfactualDTAEnvs(
        init_mol=original_drug_smile,
        init_prot=original_target,
        discount_factor=Hyperparams.discount,
        atom_types=set(atoms_),
        allow_removal=True,
        allow_no_modification=False,
        allow_bonds_between_rings=True,
        allowed_ring_sizes=set(Hyperparams.allowed_ring_sizes),
        max_steps=Hyperparams.max_steps_per_episode,
        model_to_explain=model_to_explain,
        original_molecule=original_drug,
        original_target=original_target,
        target_aff=original_target_aff,
        original_prediction=pred_aff,
        similarity_measure="neural_encoding",
        Hyperparams=Hyperparams,
        device=device,
        cof=cof
    )
    env = MultiAgentDTAEnvs(world,
                            observation_callback=world.observation,
                            reward_callback=world.reward,
                            done_callback=world.done,
                            reset_callback=world.reset,
                            Hyperparams=Hyperparams)
    return env
