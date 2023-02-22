# Last-Iterate Convergence with Full and Noisy Feedback in Two-Player Zero-Sum Games
Code for reproducing results in the paper "[Last-Iterate Convergence with Full and Noisy Feedback in Two-Player Zero-Sum Games](https://arxiv.org/abs/2208.09855)".

## About
This paper proposes Mutation-Driven Multiplicative Weights Update (M2WU) for learning an equilibrium in two-player zero-sum normal-form games and proves that it exhibits the last-iterate convergence property in both full and noisy feedback settings.
In the former, players observe their exact gradient vectors of the utility functions.
In the latter, they only observe the noisy gradient vectors.
Even the celebrated Multiplicative Weights Update (MWU) and Optimistic MWU (OMWU) algorithms may not converge to a Nash equilibrium with noisy feedback.
On the contrary, M2WU exhibits the last-iterate convergence to a stationary point near a Nash equilibrium in both feedback settings.
We then prove that it converges to an exact Nash equilibrium by iteratively adapting the mutation term.
We empirically confirm that M2WU outperforms MWU and OMWU in exploitability and convergence rates.

## Installation
This code is written in Python 3.
To install the required dependencies, execute the following command:
```bash
$ pip install -r requirements.txt
```

### For Docker User
Build the container:
```bash
$ docker build -t m2wu .
```
After build finished, run the container:
```bash
$ docker run -it m2wu
```

## Run Experiments
In order to investigate the performance of M2WU in biased Rock-Paper-Scissors with full feedback, execute the following command:
```bash
$ python run_experiment.py --num_trials 10 --T 100000 --feedback full --algorithm m2wu --random_init_strategy --eta 0.1 --mu 0.1
$ python run_experiment.py --num_trials 10 --T 100000 --feedback full --algorithm m2wu --random_init_strategy --eta 0.1 --mu 0.1 --update_freq 100
```

To evaluate M2WU via an experiment in biased Rock-Paper-Scissors with noisy feedback, execute the following command:
```bash
$ python run_experiment.py --num_trials 10 --T 1000000 --feedback noisy --algorithm m2wu --eta 0.001 --mu 0.1
$ python run_experiment.py --num_trials 10 --T 1000000 --feedback noisy --algorithm m2wu --eta 0.001 --mu 0.5 --update_freq 20000
```
