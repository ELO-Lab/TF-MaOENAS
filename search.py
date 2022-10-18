import pickle
import oapackage
import json
import argparse
import numpy as np
import torch
from thop import profile
from tabulate import tabulate
import logging
import time
import os
import sys

from pymoo.optimize import minimize
from pymoo.core.problem import Problem 
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.algorithms.moo.nsga2 import NSGA2

from foresight.pruners import predictive
from foresight.weight_initializers import init_net
from foresight.dataset import get_cifar_dataloaders
from foresight.models.nasbench2 import get_model_from_arch_str
from utils import *
from IGD import *
from nats_bench import create



def parse_arguments():
    parser = argparse.ArgumentParser("Many-objetive NAS with Synflow")
    parser.add_argument('--method', type=str, default='TF-MaOENAS', help='method [TF-MOENAS, TF-MaOENAS, MOENAS, MaOENAS]')
    parser.add_argument('--complexity_obj', type=str, default='flops', help='Complexity objective used for MOENAS and TF-MOENAS [flops, params, latency, macs]')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--pop_size', type=int, default=20, help='population size of networks')
    parser.add_argument('--n_gens', type=int, default=50, help='number of generations')

    parser.add_argument('--init_w_type', type=str, default='none', help='weight initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--init_b_type', type=str, default='none', help='bias initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use [cifar10, cifar100, imagenet]')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to work on')
    parser.add_argument('--num_data_workers', type=int, default=2, help='number of workers for dataloaders')
    parser.add_argument('--dataload', type=str, default='random', help='random or grasp supported')
    parser.add_argument('--dataload_info', type=int, default=1, help='number of batches to use for random dataload or number of samples per class for grasp dataload')

    args = parser.parse_args()
    args.save = 'experiment/search-{}-{}-{}'.format(args.method, args.dataset, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(args.save)
    args.device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")

    return args



def evaluate_arch(self, ind, dataset, measure):
    arch_str = to_string(ind) 
    epoch = 12 if measure=='valid-accuracy' else 200
    if dataset=='imagenet': dataset='ImageNet16-120'
    if measure.startswith(("train", "val", "test")):     
        if dataset=='cifar10' and measure=='valid-accuracy': dataset='cifar10-valid' 
        
        xinfo = self.api.get_more_info(
            arch_str,
            hp=epoch,
            dataset=dataset,
            is_random=False,
        )
        res = xinfo[measure]
    elif measure in ['flops', 'latency', 'params']:
        arch_index = self.api.query_index_by_arch(arch_str)
        info = self.api.get_cost_info(arch_index, dataset)
        res = info[measure]

    elif measure in ['synflow', 'macs']:
        net = get_model_from_arch_str(arch_str, get_num_classes(self.args))
        net.to(self.args.device)
        init_net(net, self.args.init_w_type, self.args.init_b_type)
        if measure=='synflow':
            measures = predictive.find_measures(net, 
                                                self.train_loader, 
                                                (self.args.dataload, self.args.dataload_info, get_num_classes(self.args)), 
                                                self.args.device,
                                                measure_names=[measure])    
            res = measures[measure]
        elif measure=='macs':
            input = get_input(self.args, self.train_loader)
            res, _ = profile(net, inputs=(input, ), verbose=False)
    return res



class NAS(Problem):
    def __init__(self, n_var=6, n_obj=5, dataset='cifar10', xl=None, xu=None, save_dir=None, seed=0, objectives_list=None, args=None):
        super().__init__(n_var=n_var, n_obj=n_obj)
        self.xl = xl
        self.xu = xu
        self._save_dir = save_dir
        self._n_generation = 0
        self._n_evaluated = 0
        self.archive_obj = []
        self.archive_var = []
        self.seed = seed
        self.dataset = dataset
        self.datasets = ['cifar10', 'cifar100', 'imagenet'] if dataset=='cifar10' else [dataset]
        self.objectives_list = objectives_list
        self.pf = pickle.load(open('benchmark/pareto_front/pf.pickle', "rb"))
        self.pf_norm = pickle.load(open('benchmark/pareto_front/pf_norm.pickle', "rb"))
        self.max_min_measures = pickle.load(open('benchmark/max_min_measures.pickle', "rb"))
        self.api = create("benchmark/NATS-tss-v1_0-3ffb9-simple", 'tss', fast_mode=True, verbose=False)

        self.args = args
        if dataset=='imagenet':
            self.train_loader, self.test_loader = get_cifar_dataloaders(args.batch_size, args.batch_size, 'ImageNet16-120', args.num_data_workers, resize=None, datadir='benchmark/')
        else:
            
            self.train_loader, self.test_loader = get_cifar_dataloaders(args.batch_size, args.batch_size, args.dataset, args.num_data_workers)
            
        if n_var > 0:
            if self.xl is not None:
                if not isinstance(self.xl, np.ndarray):
                    self.xl = np.ones(n_var) * xl
                self.xl = self.xl.astype(float)

            if self.xu is not None:
                if not isinstance(self.xu, np.ndarray):
                    self.xu = np.ones(n_var) * xu
                self.xu = self.xu.astype(float)

    def _evaluate(self, x, out, *args, **kwargs):
        logging.info('Generation: {}'.format(self._n_generation))

        objs = np.full((x.shape[0], self.n_obj), np.nan)
        

        for i in range(x.shape[0]):
            for j in range(len(self.objectives_list)): 
                # all objectives assume to be MINIMIZED !!!!!
                obj = evaluate_arch(self, ind=x[i], dataset=self.dataset, measure=self.objectives_list[j])

                if 'accuracy' in self.objectives_list[j] or self.objectives_list[j]=='synflow':
                    objs[i, j] = -1 * obj
                    print(obj)
                    print(objectives_list[j])
                else: 
                    objs[i, j] = obj
            self.archive_obj, self.archive_var = archive_check(objs[i], self.archive_obj, self.archive_var, x[i])
            self._n_evaluated += 1

            igd_dict, igd_norm_dict = calc_IGD(self, x=x, objs=objs)

        igd_norm_list = []
        for dataset in self.datasets:
            igd_temp = list(igd_norm_dict[dataset].values())
            igd_temp.insert(0, dataset)
            igd_norm_list.append(igd_temp)
            
        headers = ['']
        for j in range(1, len(self.objectives_list)): 
            headers.append('test accuracy - ' + self.objectives_list[j])
        logging.info(tabulate(igd_norm_list, headers=headers, tablefmt="grid"))

        self._n_generation += 1
        out["F"] = objs

if __name__ == '__main__':
    args = parse_arguments()
    torch.manual_seed(args.seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    
    if args.method=='MOENAS':
        objectives_list = ['valid-accuracy', args.complexity_obj]
    elif args.method=='TF-MOENAS':
        objectives_list = ['synflow', args.complexity_obj]
    elif args.method=='MaOENAS':
        objectives_list = ['valid-accuracy', 'flops', 'params', 'latency', 'macs']
    elif args.method=='TF-MaOENAS':
        objectives_list = ['synflow', 'flops', 'params', 'latency', 'macs']
    
    logging.info('seed: {}'.format(args.seed))
    logging.info('objectives: {}'.format(objectives_list))
    
    n_obj = len(objectives_list)
    n_var = 6 #NATS-Bench
    problem = NAS(objectives_list=objectives_list, n_var=n_var,
                n_obj=n_obj, dataset=args.dataset, 
                xl=0, xu=4, save_dir=args.save, seed=args.seed, args=args)

    algorithm = NSGA2(pop_size=args.pop_size,
                    sampling=IntegerRandomSampling(),
                    crossover=TwoPointCrossover(prob=0.9),
                    mutation=PolynomialMutation(prob=1.0/n_var, eta=1.0, repair=RoundingRepair()),
                    eliminate_duplicates=True)

    stop_criteria = ('n_gen', args.n_gens)

    results = minimize(
        problem = problem,
        algorithm = algorithm, 
        seed = args.seed,
        termination = stop_criteria
    )