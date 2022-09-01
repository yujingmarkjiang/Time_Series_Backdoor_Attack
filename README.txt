All 13 datasets used in this paper have been included.


To run the clean model:
python main.py run_baseline

To run the vanilla backdoor method:
python main.py run_backdoor vanilla

To run the static noise backdoor method:
python main.py run_backdoor powerline

To run our proposed TSBA:
python main.py run_backdoor generator

To test the generator from trained TSBA model:
python main.py run_backdoor generative_test