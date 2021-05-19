# Counterfactual Explanation with Multi-Agent Reinforcement Learning for Drug Target Prediction

**Motivation**: Several accurate deep learning models have been proposed to predict drug-target affinity(DTA). However, all of these models are black box hence are difficult to interpret and verify its result, andthus risking acceptance. Explanation is necessary to allow the DTA model more trustworthy. Explanationwith counterfactual provides human-understandable examples. Most counterfactual explanation methodsonly operate on single input data, which are in tabular or continuous forms. In contrast, the DTA model hastwo discrete inputs. It is challenging for the counterfactual generation framework to optimize both discreteinputs at the same time. In addition, the interaction between sub-structure of two inputs, drug functionalgroups and protein residues, is an important factor in the DTA model prediction. Explanation based onsubstructure interaction allows domain experts to verify the binding mechanism used by DTA model in itsprediction.

**Results**: We propose a multi-agent reinforcement learning
framework, Multi-Agent Counterfactual Drug-target binding Affinity
(MACDA), to generate counterfactual explanations for the drug-protein
complex. Our proposed framework provides human-interpretable counterfactual
instances while optimizing both the input drug and target for counterfactual
generation at the same time. The result on the Davis dataset shows
the advantages of the proposed MACDA framework compared with previous
works. 

#Usage

First, create conda environment and active the :
```
conda env create --file=environment.yml
conda activate torchgeo_161
```

Second, preprocess data. You can preprocess your own data from a new csv file following the example csv file.
```
python create_data.py davis_train_ABL1.csv
```

Finally, to train and generate counterfactual example run:

```
python run_macda.py 
--preprocessfile [your preprocessed file created in the previous step]
--gpu [which gpu you will run]
--n_episodes [the number of episode to run/ default = 10000]
--store_result_dir [folder to save the the result]
```

The counterfactual examples are placed in the 'runs/[store_result_dir]/counterfacts/'

#Citation
If you find this work useful, please cite our paper:
```
@article{nguyen2021counterfactual,
  title={Counterfactual Explanation with Multi-Agent Reinforcement Learning for Drug Target Prediction},
  author={Nguyen, Tri Minh and Quinn, Thomas P and Nguyen, Thin and Tran, Truyen},
  journal={arXiv preprint arXiv:2103.12983},
  year={2021}
}
```