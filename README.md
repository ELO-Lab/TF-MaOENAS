# Training-Free Multi-Objective and Many-Objective Evolutionary Neural Architecture Search with Synaptic Flow
[![MIT licensed](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE.md)

An Vo, Tan Ngoc Pham, Van Bich Nguyen, Ngoc Hoang Luong

In SoICT 2022.
## Setup
- Clone this repository
- Install packages
```
$ pip install -r requirements.txt
```
- Download [NATS-bench](https://drive.google.com/drive/folders/17S2Xg_rVkUul4KuJdq0WaWoUuDbo8ZKB), put it in the `benchmark` folder and follow instructions [here](https://github.com/D-X-Y/NATS-Bench)
## Usage

```shell
python search.py --method <method_name>
               --complexity_obj <complexity_objective> --pop_size <population_size>
               --n_gens <num_of_generations> --dataset <dataset_name> --seed <random_seed> 
```

Please see details in `search.py.`

```shell
# MOENAS
python search.py --method MOENAS --complexity_obj flops

# TF-MOENAS
python search.py --method TF-MOENAS --complexity_obj flops

# MaOENAS
python search.py --method MaOENAS

# TF-MOENAS
python search.py --method TF-MaOENAS 
```


## Acknowledgement
Our source code is inspired by:
- [pymoo: Multi-objective Optimization in Python](https://github.com/anyoptimization/pymoo)
- [NSGA-Net: Neural Architecture Search using Multi-Objective Genetic Algorithm](https://github.com/ianwhale/nsga-net)
- [Zero-Cost Proxies for Lightweight NAS](https://github.com/SamsungLabs/zero-cost-nas)
- [NATS-Bench: Benchmarking NAS Algorithms for Architecture Topology and Size](https://github.com/D-X-Y/NATS-Bench)