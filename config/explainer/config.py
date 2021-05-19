from collections import namedtuple
from os import path


_Hyperparams = namedtuple(
    'Hyperparams',
    [
        'sample',
        'preprocessfile',
        'gpu',
        'seed',
        'dataset',
        'store_result_dir',
        'optimizer',
        'polyak',
        'max_steps_per_episode',
        'allowed_ring_sizes',
        'replay_buffer_size',
        'lr',
        'gamma',
        'fingerprint_radius',
        'fingerprint_length',
        'discount',
        'n_episodes',
        'batch_size',
        'num_updates_per_it',
        'update_interval',
        'num_counterfactuals'
    ]
)

_Path = namedtuple(
    'Path',
    [
        'data',
        'counterfacts',
        'drawings'
    ]
)

Hyperparams = None


def Args():

    if Hyperparams is not None:
        return Hyperparams

    import argparse as ap
    parser = ap.ArgumentParser(description='Explainer Hyperparams')
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--dataset", default="davis_graphdta", help="Name of environment")
    parser.add_argument("--preprocessfile", default="davis_train_ABL1", help="Preprocessed file")
    parser.add_argument("--store_result_dir", default="test_ABL1", help="Result storage dir")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning Rate")
    parser.add_argument('--optimizer', choices=['Adam', 'SGD', 'Adamax'],
                        default='Adam')
    parser.add_argument('--polyak', type=float, default=0.995)
    parser.add_argument('--max_steps_per_episode', type=int, default=1)

    parser.add_argument('--replay_buffer_size', type=int, default=1000000)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--fingerprint_radius', type=int, default=2)
    parser.add_argument('--fingerprint_length', type=int, default=4096)
    parser.add_argument('--discount', type=bool, default=0.9)
    parser.add_argument('--n_episodes', type=int, default=10000)
    parser.add_argument('--num_counterfactuals', type=int, default=15)

    args = parser.parse_args()

    return _Hyperparams(
        sample=args.sample,
        store_result_dir=args.store_result_dir,
        dataset=args.dataset,
        gpu=args.gpu,
        seed=args.seed,
        optimizer=args.optimizer,
        polyak=args.polyak,
        max_steps_per_episode=args.max_steps_per_episode,
        allowed_ring_sizes=[5, 6],
        replay_buffer_size=args.replay_buffer_size,
        lr=args.lr,
        gamma=args.gamma,
        fingerprint_radius=args.fingerprint_radius,
        fingerprint_length=args.fingerprint_length,
        discount=args.discount,
        n_episodes=args.n_episodes,
        batch_size=1,
        num_updates_per_it=1,
        update_interval=1,
        num_counterfactuals=args.num_counterfactuals,
        preprocessfile=args.preprocessfile
    )


Log = print

_BasePath = path.normpath(path.join(
    path.dirname(path.realpath(__file__)),
    '..',
    '..'
))

Path = _Path(
    data=lambda x: path.join(_BasePath, 'data', x),
    counterfacts=lambda x, d: path.join(_BasePath, 'counterfacts', 'files', d, x),
    drawings=lambda x, d: path.join(_BasePath, 'counterfacts', 'drawings', d, x),
)
