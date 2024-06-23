# ResearchProject
## Install
First, clone this repository by navigate to the root of project, and perform:
```
pip3 install -e .
```

## Run DICE Algorithms
First, create datasets using the policy trained above (For the FrozenLake Environment) using 1 as the random seed:
```
for alpha in {0.0,1.0}; do python3 scripts/create_dataset.py --save_dir=./tests/testdata --load_dir=./tests/testdata/frozenlake --env_name=frozenlake --num_trajectory=400 --max_trajectory_length=250 --alpha=$alpha --tabular_obs=0 --seed=1; done
```

Then, run the following command to extract the Q-values:
```
python3 scripts/run_neural_dice.py --save_dir=./tests/testdata --load_dir=./tests/testdata --env_name=frozenlake --num_trajectory=400 --max_trajectory_length=250 --alpha=0.0 --tabular_obs=0 --seed=1
```

Lastly, run the QLearning_FrozenLake, which automatically plots the graph on the average cumulative rewards.