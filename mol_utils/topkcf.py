from config.explainer import Path, Args

class TopKCounterfactualsDTA:
    Leaderboard = None
    K = 5
    save_dir = None

    @staticmethod
    def init(original, index, save_dir, k=10):
        TopKCounterfactualsDTA.Leaderboard = None
        TopKCounterfactualsDTA.save_dir = save_dir
        TopKCounterfactualsDTA.K = k

        if TopKCounterfactualsDTA.Leaderboard is None:
            TopKCounterfactualsDTA.Leaderboard = {
                'original': original,
                'index': index,
                'counterfacts': [
                    {
                        'smiles': '',
                        'protein': '',
                        'drug_reward': 0,
                        'protein_reward': 0,
                        'loss': 0,
                        'gain': 0,
                        'drug sim': 0,
                        'prot sim': 0,
                        'mutate position': -1
                    }
                    for _ in range(k)
                ]
            }

    @staticmethod
    def insert(counterfact):

        Leaderboard = TopKCounterfactualsDTA.Leaderboard
        K = TopKCounterfactualsDTA.K

        if any(
            (x['protein'] == counterfact['protein'] and x['smiles'] == counterfact['smiles'])
            for x in Leaderboard['counterfacts']
        ):
            return

        Leaderboard['counterfacts'].extend([counterfact])
        Leaderboard['counterfacts'].sort(
            reverse=True,
            key=lambda x: x['drug_reward'] + x['protein_reward']
        )
        Leaderboard['counterfacts'] = Leaderboard['counterfacts'][:K]

        TopKCounterfactualsDTA._dump()

    @staticmethod
    def _dump():
        import json
        with open(TopKCounterfactualsDTA.save_dir + '/' + str(TopKCounterfactualsDTA.Leaderboard['index']) + '.json', 'w') as f:
            json.dump(TopKCounterfactualsDTA.Leaderboard, f, indent=2)

