# SMiR
## Preparation

**Our computer configuration**

SMiR was implemented in **Python 3.8.10** and **PyTorch 1.12.1**. We conducted our experiments on a 64-bit Ubuntu 20.04 server with an Intel Xeon Gold 6330 CPU, an RTX 3090 GPU, and 80 GB of memory. 

**Dependencies**

Install in the root directory:

```bash
pip install -r requirements.txt
pip install torch==1.12.1
pip install tqdm
pip install matplotlib
pip install wandb
```



##  How to run



__Example__

```bash
python smir_main.py \ 
	--rq "rq1"
    --dataset "cifar10" 
    --model "simple_cm"
    --is_loc "True"
    --only_loc "False"
    --method "smir"
```

`dataset` denotes the chosen dataset, choice in `["fashion_mnist", "cifar_10", "svhn", "NEU-CLS-64", "APTOS2019"]`

`model` denotes the chosen model, choice in `["simple_fm", "simple_cm", "C10_CNN1", "NEU-simple_svhn-64", "SVHN_CNN1", "NEU-CLS-64_CNN", "APTOS2019_ResNet18"]`



## Baselines
### Arachne
We ran Arachne on a server with RTX 2080Ti supported TensorFlow 1.15.5.

### Apricot

Unofficial implementation following **[Arachne]**.



## Citation



*We will update the remaining parts as soon as possible.*.
