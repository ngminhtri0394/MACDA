import os
from pathlib import Path
import argparse
import numpy as np
import torch
from rdkit import Chem
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import mol_utils
from algorithms.attention_sac import AttentionSAC
from config.explainer import Args, Log
from mol_utils import preprocess, seq_cat
from utils.buffer import ReplayBuffer
from utils.env_wrappers import DummyVecEnv
from utils.make_env import make_env
from tqdm import tqdm, trange

def make_parallel_env(original_drug_smile, original_target, Hyperparams, atoms_, model_to_explain, original_drug,
                      original_target_aff, pred_aff, device, cof):
    def get_env_fn(rank):
        def init_env():
            env = make_env(original_drug_smile, original_target, Hyperparams, atoms_, model_to_explain, original_drug,
                           original_target_aff, pred_aff, device, cof)
            return env

        return init_env

    return DummyVecEnv([get_env_fn(0)])


def run(config):
    device = torch.device('cuda:'+str(config.gpu) if torch.cuda.is_available() else 'cpu')
    model_dir = Path('./runs') / config.store_result_dir

    train_loader, train_drugs, train_Y = preprocess(config.dataset, config)

    print("number of data")
    print(len(train_loader))
    for it, original_pair in enumerate(train_loader):
        if not model_dir.exists():
            run_num = 1
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                             model_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                run_num = 1
            else:
                run_num = max(exst_run_nums) + 1
        curr_run = 'run%i' % run_num
        run_dir = model_dir / curr_run
        log_dir = run_dir / 'logs'
        os.makedirs(log_dir)
        logger = SummaryWriter(str(log_dir))

        torch.manual_seed(run_num)
        np.random.seed(run_num)

        print('Run pair number ', str(it))
        Hyperparams = Args()
        BasePath = './runs/' + config.store_result_dir
        writer = SummaryWriter(BasePath + '/plots')

        original_drug_smile = train_drugs[it]
        original_target_aff = train_Y[it]
        original_drug = original_pair
        original_target = original_pair.target[0]

        print('Original target:')
        print(original_target)
        print('Original molecule:')
        print(original_drug_smile)

        model_to_explain = mol_utils.get_graphdta_dgn().to(device)
        pred_aff, drug_original_encoding, prot_original_encoding = model_to_explain(original_drug.to(device),
                                                                                    seq_cat(original_target).to(device))
        atoms_ = np.unique(
            [x.GetSymbol() for x in Chem.MolFromSmiles(original_drug_smile).GetAtoms()]
        )
        cof = [1.0, 0.05, 0.01, 0.05]
        env = make_parallel_env(original_drug_smile,
                                original_target,
                                Hyperparams,
                                atoms_,
                                model_to_explain,
                                original_drug,
                                original_target_aff,
                                pred_aff,
                                device,
                                cof)
        model = AttentionSAC.init_from_env(env,
                                           tau=config.tau,
                                           pi_lr=config.pi_lr,
                                           q_lr=config.q_lr,
                                           gamma=config.gamma,
                                           pol_hidden_dim=config.pol_hidden_dim,
                                           critic_hidden_dim=config.critic_hidden_dim,
                                           attend_heads=config.attend_heads,
                                           reward_scale=config.reward_scale)
        replay_buffer = ReplayBuffer(config.buffer_length, model.nagents,
                                     [obsp[0] for obsp in env.observation_space],
                                     [acsp for acsp in env.action_space])

        if not os.path.isdir(BasePath + "/counterfacts"):
            os.makedirs(BasePath + "/counterfacts")
        mol_utils.TopKCounterfactualsDTA.init(original_drug_smile, it, BasePath + "/counterfacts")

        t = 0
        episode_length = 1
        trg = trange(0, config.n_episodes, config.n_rollout_threads)
        for ep_i in trg:
            obs = env.reset()
            model.prep_rollouts(device='cpu')

            for et_i in range(episode_length):
                # rearrange observations to be per agent, and convert to torch Variable
                torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                      requires_grad=False)
                             for i in range(model.nagents)]
                # get actions as torch Variables
                torch_agent_actions = model.step(torch_obs, explore=True)
                # convert actions to numpy arrays
                agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
                # rearrange actions to be per environment
                actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
                next_obs, results, dones, action_drug, action_prot = env.step(actions)
                drug_reward, loss_, gain, drug_sim, prot_sim, qed = results[0][0]
                prot_reward, loss_, gain, drug_sim, prot_sim, qed = results[0][1]

                writer.add_scalar('DTA/Reward', drug_reward, ep_i)
                writer.add_scalar('DTA/Distance', loss_, ep_i)
                writer.add_scalar('DTA/Drug Similarity', drug_sim, ep_i)
                writer.add_scalar('DTA/Drug QED', qed, ep_i)
                writer.add_scalar('DTA/Protein Similarity', prot_sim, ep_i)

                pair_reward = []
                pair_reward.append(drug_reward)
                pair_reward.append(prot_reward)
                rewards = np.array([pair_reward])
                replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
                obs = next_obs
                t += 1
                if (len(replay_buffer) >= config.batch_size and
                        (t % config.steps_per_update) < 1):
                    if config.use_gpu:
                        model.prep_training(device='gpu')
                    else:
                        model.prep_training(device='cpu')
                    for u_i in range(config.num_updates):
                        sample = replay_buffer.sample(config.batch_size,
                                                      to_gpu=config.use_gpu)
                        model.update_critic(sample, logger=logger)
                        model.update_policies(sample, logger=logger)
                        model.update_all_targets()
                    model.prep_rollouts(device='cpu')
                if np.all(dones == True):
                    mutate_position = [i for i in range(len(original_target)) if original_target[i] != action_prot[i]]
                    trg.set_postfix(Reward=drug_reward, DrugSim=drug_sim, TargetSim=prot_sim, SMILES=action_drug, TargetMutatePosition=mutate_position, refresh=True)
                    mol_utils.TopKCounterfactualsDTA.insert({
                        'smiles': action_drug,
                        'protein': action_prot,
                        'drug_reward': drug_reward,
                        'protein_reward': prot_reward,
                        'loss': loss_,
                        'gain': gain,
                        'drug sim': drug_sim,
                        'drug qed': qed,
                        'prot sim': prot_sim,
                        'mutate position': mutate_position
                    })
            ep_rews = replay_buffer.get_average_rewards(
                episode_length * 1)
            for a_i, a_ep_rew in enumerate(ep_rews):
                logger.add_scalar('agent%i/mean_episode_rewards' % a_i,
                                  a_ep_rew * episode_length, ep_i)

            if ep_i % config.save_interval < config.n_rollout_threads:
                model.prep_rollouts(device='cpu')
                os.makedirs(run_dir / 'incremental', exist_ok=True)
                model.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
                model.save(run_dir / 'model.pt')

        model.save(run_dir / 'model.pt')
        env.close()
        logger.export_scalars_to_json(str(log_dir / 'summary.json'))
        logger.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessfile", default="davis_train_ABL1", help="Preprocessed file")
    parser.add_argument("--dataset", default="davis_graphdta", help="Name of environment")
    parser.add_argument("--store_result_dir", default="test_ABL1", help="Result storage dir")
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=10000, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--num_updates", default=4, type=int,
                        help="Number of updates per update cycle")
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for training")
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--pol_hidden_dim", default=128, type=int)
    parser.add_argument("--critic_hidden_dim", default=128, type=int)
    parser.add_argument("--attend_heads", default=4, type=int)
    parser.add_argument("--pi_lr", default=0.001, type=float)
    parser.add_argument("--q_lr", default=0.001, type=float)
    parser.add_argument("--tau", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--reward_scale", default=100., type=float)
    parser.add_argument("--use_gpu", action='store_true')

    config = parser.parse_args()

    run(config)
