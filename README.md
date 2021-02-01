# Counterfactual Explanation with Multi-Agent Reinforcement Learning for Drug Target Prediction

Motivation: Several accurate deep learning models have been
proposed to predict drug-target affinity (DTA). However, all of these
models are black box hence are difficult to interpret and verify its
result, and thus risking acceptance. Explanation is necessary to allow
the DTA model more trustworthy. Explanation with counterfactual provides
human-understandable examples. Most counterfactual explanation methods
only operate on single input data, which are in tabular or continuous
forms. In contrast, the DTA model has two discrete inputs. It is challenging
for the counterfactual generation framework to optimize both discrete
inputs at the same time.

Results: We propose a multi-agent reinforcement learning
framework, Multi-Agent Counterfactual Drug-target binding Affinity
(MACDA), to generate counterfactual explanations for the drug-protein
complex. Our proposed framework provides human-interpretable counterfactual
instances while optimizing both the input drug and target for counterfactual
generation at the same time. The result on the Davis dataset shows
the advantages of the proposed MACDA framework compared with previous
works. 
