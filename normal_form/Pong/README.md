# Environment

Please install the atari environment before executing any program. We use the `Pong-v0` environment for our experiment.

# Code Structure 

We provide a pre-trained target agent in `./ppo_test/baseline/Pong-v0_+0.896_12150.dat` folder.

- `ppo.py`: main script to train a mask network.
- `model.py`: implement the network of DRL agents.
- `test.py`: test the target agent's performance and the masknet's performance.
- `select_critical_steps.py`: select critical steps based on explanation.
- `replay.py`: replay the trajectories and replace the actions in the selected criticap steps with random actions.
- `random_replace.py`: randomly select critical steps and replace the actions in the selected criticap steps with random actions.
- `fidelity_score.py`: calculate the fidelity score.
- `split_mask_video.py`: split the recording into frames.
- `attack.py`: launch adversarial attacks based on explanation.
- `generate_retraining_data.py`: generate trajectories with the corresponding explanation for retraining.


# StateMask

- Run `ppo.py` to train the masknet. To train from scratch, set `TRANSFER_LEARNING=False` in `ppo.py`. We give one trained masknet model in `./ppo_test/checkpoints/Pong-v0_+0.855_19700.dat`. It can be used to calculate the fidelity test score or loaded as a pre-trained masknet model (set `TRANSFER_LEARNING=True` in `ppo.py`).
- Run `test.py` to test the performance of the target agent and the masknet.
- To do fidelity tests, execute the following programs:
- - `python test.py` to generate trajectories for explanation
  - `python select_critical_steps.py ` to select most important time steps
  - `python replay.py` to randomize actions in the identified critical steps
  - `python fidelity_score.py` to calculate the fidelity score.
- We provide the code for applications such as attack and retraining in `attack.py` and `retrain.py`.
