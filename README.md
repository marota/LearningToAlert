# LearningToAlert
Train an alert module for a grid2op agent, which sends alerts if the grid is likely to collaps due to agent actions.

Any agent compatible with the [grid2op framework](https://github.com/rte-france/Grid2Op) can be specified, and is ran to collect data. Contingencies are simulated, and the policy is ran for a specified number of steps. Data on wether the agent managed to save the grid or not, along with failure timestep, is registered.

The collected data is then used to create a learning target Y for a supervised training model. The input data X is extracted from a list of grid observations.

At evaluation, the trained model is used to raise alerts, and should be attached to the agent used to collect the data.

## Usage
1. Gather data for supervised learning:
  - Open `data_collector.py` and configure 'main()' to load your desired agent (grid controller)
  - Run the file
  - :bulb: `data_collector.py` can also be run in parallel, check out `parallel_data_collection.py` if interested
2. Pre-process data and create train-test set:
  - Open `pre_process_data.ipynb` and run the relevant steps
  - :warning: You must run the code which creates a folder containing:
    - X_train.csv
    - Y_train.csv
    - X_test.csv
    - Y_test.csv
3. Train model to predict grid survival:
  - Open either `models_keras.py` (MLP models) or `models_xgboost.py` (boosted tree models)
  - In 'main()', update path to the folder containing your training and test data
  - Optionally configure model parameters, training settings etc
  - Run file
4. Evaluate the performance of your alert module:
  - Open `eval.ipynb`
  - Run and save some scenarios
  - Load all results or perform case study
  - :bulb: Running scenarios can take some time. One can run EvalRunner directly (see `eval_runner.py`)

## Installation
1. Clone respository
2. Create a virtual environment
  - `python3 -m venv venv_alert`
4. Activate venv
  - `source venv_alert/bin/activate`
5. Cd to repository
  - `cd l2rpn_2023_alert`
7. Install requirements
  - `pip install -r requirements.txt`

Done!

## License
Copyright (c) [2023-2024], RTE (https://www.rte-france.com)

See [LICENSE file](https://github.com/rte-france/LearningToAlert/blob/main/LICENSE)

This agent code is part of L2RPN (Learning To run a Power Network) Open-Science initiative, which aims at accelerating the development of AI solutions for power grid operations management.
