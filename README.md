# TSBA: Time Series Backdoor Attack

This is the code for the [SaTML'23 paper](https://arxiv.org/pdf/2211.07915.pdf) "Backdoor Attacks on Time Series: A Generative Approach" by Yujing Jiang, Xingjun Ma, Sarah Monazam Erfani, and James Bailey.

## Prerequisites
* Python (3.9.7)
* Pytorch (1.10.0)
* CUDA (with 4 GPUs)

## How to run

To run the clean model:
```
python main.py run_baseline
```

To run the vanilla backdoor method:
```
python main.py run_backdoor vanilla
```

To run the static noise backdoor method:
```
python main.py run_backdoor powerline
```

To run our proposed TSBA:
```
python main.py run_backdoor generator
```

To test the generator from trained TSBA model:
```
python main.py run_backdoor generative_test
```

## Reference
For technical details and full experimental results, please check [the paper](https://arxiv.org/pdf/2211.07915.pdf).
```
@inproceedings{xxxxx,
  title={Backdoor Attacks on Time Series: A Generative Approach},
  author={Jiang, Yujing and Ma, Xingjun and Erfani, Sarah Monazam and Bailey, James},
}
```



