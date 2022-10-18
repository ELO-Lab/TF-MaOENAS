import oapackage
import numpy as np
from search import evaluate_arch
from utils import *

from pymoo.indicators.igd import IGD


def archive_check(ind_obj, archive_obj, archive_var, ind_var) -> list:
    def is_dominated(ind_obj, ind_archive) -> bool:
        condition_forall = True
        condition_exists = False

        for measure in range(len(ind_obj)):
            if ind_obj[measure] > ind_archive[measure]: 
                condition_forall = False
                break
        for measure in range(len(ind_obj)):
            if ind_obj[measure] < ind_archive[measure]:
                condition_exists = True
                break
        return condition_forall and condition_exists
    
    idx_indobj_dominated_by_archive = []

    for idx_archive in range(len(archive_obj)):
        if is_dominated(ind_obj, archive_obj[idx_archive]):
            idx_indobj_dominated_by_archive.append(idx_archive)
        
        if is_dominated(archive_obj[idx_archive], ind_obj):
            return archive_obj, archive_var

    archive_obj_result = [ind_obj]
    archive_var_result = [ind_var]
    for idx in range(len(archive_obj)):
        if idx not in idx_indobj_dominated_by_archive:
            archive_obj_result.append(archive_obj[idx])
            archive_var_result.append(archive_var[idx])

    return archive_obj_result, archive_var_result

def remove_ind_dominated(archive_eval):
    pareto=oapackage.ParetoDoubleLong()

    archive_eval = np.array(archive_eval).T
    archive_eval[0] = -archive_eval[0]

    for ii in range(0, archive_eval.shape[1]):
        w=oapackage.doubleVector((archive_eval[0,ii], archive_eval[1,ii]))
        pareto.addvalue(w, ii)

    lst=pareto.allindices() 
    archive_eval = archive_eval[:,lst]
    archive_eval[0] = -archive_eval[0]
    archive_eval = archive_eval.T 
    
    return archive_eval

def calc_IGD(self, x, objs):
    archive_eval, archive_eval_norm = {}, {}
    igd_dict, igd_norm_dict = {}, {}

    for dataset in self.datasets:
        archive_eval[dataset], archive_eval_norm[dataset] = {}, {}
        igd_dict[dataset], igd_norm_dict[dataset] = {}, {}
        for j in range(1, len(self.objectives_list)):
            archive_eval[dataset][f'testacc_{self.objectives_list[j]}'] = []        
            archive_eval_norm[dataset][f'testacc_{self.objectives_list[j]}'] = []
            for ind in self.archive_var:
                accuracy = evaluate_arch(self, ind=ind, dataset=dataset, measure='test-accuracy')
                complexity = evaluate_arch(self, ind=ind, dataset=dataset, measure=self.objectives_list[j])
                archive_eval[dataset][f'testacc_{self.objectives_list[j]}'].append((complexity, accuracy))                

            archive_eval[dataset][f'testacc_{self.objectives_list[j]}'] = remove_ind_dominated(archive_eval[dataset][f'testacc_{self.objectives_list[j]}'])
            archive_eval_norm[dataset][f'testacc_{self.objectives_list[j]}'] = archive_eval[dataset][f'testacc_{self.objectives_list[j]}'].copy()
            archive_eval_norm[dataset][f'testacc_{self.objectives_list[j]}'][:, 0] = (archive_eval_norm[dataset][f'testacc_{self.objectives_list[j]}'][:, 0] - self.max_min_measures[dataset][f'{self.objectives_list[j]}_min']) / (self.max_min_measures[dataset][f'{self.objectives_list[j]}_max'] - self.max_min_measures[dataset][f'{self.objectives_list[j]}_min'])
            archive_eval_norm[dataset][f'testacc_{self.objectives_list[j]}'][:, 1] = archive_eval_norm[dataset][f'testacc_{self.objectives_list[j]}'][:, 1] / 100
            
            

            get_igd = IGD(self.pf[dataset][f'testacc_{self.objectives_list[j]}'])
            get_igd_norm = IGD(self.pf_norm[dataset][f'testacc_{self.objectives_list[j]}'])

            igd = get_igd(archive_eval[self.dataset][f'testacc_{self.objectives_list[j]}'])
            igd_norm = get_igd_norm(archive_eval_norm[self.dataset][f'testacc_{self.objectives_list[j]}'])

            igd_dict[dataset][f'testacc_{self.objectives_list[j]}'] = igd
            igd_norm_dict[dataset][f'testacc_{self.objectives_list[j]}'] = igd_norm
    
    return igd_dict, igd_norm_dict



  