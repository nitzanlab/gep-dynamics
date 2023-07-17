# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Running the comparator between developmental stages in Zepp et. al.

# %%
# %%time
# %load_ext autoreload
# %autoreload 2

import os
import warnings


import pandas as pd
import scanpy as sc

from sklearn.exceptions import ConvergenceWarning

from gepdynamics import _utils
from gepdynamics import comparator

# Move to the project's home directory, as defined in _constants
_utils.cd_proj_home()
print(os.getcwd())

# %%
# %%time
results_dir = _utils.set_dir('results_zepp')

split_adatas_dir = results_dir.joinpath('split_adatas')

stages = ['E12', 'E15', 'E17', 'P3', 'P7', 'P15', 'P42']

split_adatas = {stage: sc.read_h5ad(split_adatas_dir.joinpath(f'{stage}_GEPs.h5ad')) for stage in stages}


# %%
for stage in stages:
    tmp = split_adatas[stage]
    # pd.Categorical(adata.obs['timesimple'].map(
    #    dict(zip(pd.Categorical(adata.obs['timesimple']).categories, adata.uns['timesimple_colors']))))
    field_1 = 'compartment'
    field_2 = 'celltype'
    tmp.obsm['row_colors'] = pd.concat([
        tmp.obs[field_1].map(tmp.uns[f'{field_1}_colors_dict']),
        tmp.obs[field_2].map(tmp.uns[f'{field_2}_colors_dict'])], axis=1)


# %%
# %%time

pairs = [(stages[i], stages[i + 1]) for i in range(len(stages) -1)]

for stage_a, stage_b in pairs:
    comparison_dir = _utils.set_dir(results_dir.joinpath(f"{stage_a}_{stage_b}"))
    
    adata_a = split_adatas[stage_a]
    adata_b = split_adatas[stage_b]
    
    if os.path.exists(comparison_dir.joinpath('comparator.npz')):
        cmp = comparator.Comparator.load_from_file(comparison_dir.joinpath('comparator.npz'), adata_a, adata_b)
    else:
        cmp = comparator.Comparator(adata_a, adata_a.obsm['usages'], adata_b, comparison_dir, 'torchnmf', device='cuda', max_nmf_iter=1000, verbosity=1)
        
        print('decomposing')
        cmp.extract_geps_on_jointly_hvgs()
        cmp.decompose_b(repeats = 20)
    
    cmp.examine_adata_a_decomposition_on_jointly_hvgs()
    
    cmp.print_errors()
    cmp.examine_adata_b_decompositions()
    cmp.plot_decomposition_comparisons()
    
    cmp.calculate_fingerprints()
    
    print('running GSEA')
    cmp.run_gsea(gprofiler_kwargs=dict(organism='mmusculus', sources=['GO:BP', 'WP', 'REAC', 'KEGG', 'TF', 'HP']))

    cmp.save_to_file(comparison_dir.joinpath('comparator.npz'))

    

# %%
for stage_a, stage_b in pairs:
    comparison_dir = results_dir.joinpath(f"{stage_a}_{stage_b}", 'comparator.npz')
    
    adata_a = split_adatas[stage_a]
    adata_b = split_adatas[stage_b]
    
    cmp = comparator.Comparator.load_from_file(comparison_dir, adata_a, adata_b)
    break


# %% [markdown] tags=[]
# ### Epithelial only comparator objects

# %%
# %%time

epi_split_adatas_dir = results_dir.joinpath('epi_split_adatas')

stages = ['E12', 'E15', 'E17', 'P3', 'P7', 'P15', 'P42']

epi_split_adatas = {stage: sc.read_h5ad(epi_split_adatas_dir.joinpath(f'epi_{stage}_GEPs.h5ad')) for stage in stages}


# %%
for stage in stages:
    tmp = epi_split_adatas[stage]

    field_1 = 'compartment'
    field_2 = 'celltype'
    tmp.obsm['row_colors'] = pd.concat([
        tmp.obs[field_1].map(tmp.uns[f'{field_1}_colors_dict']),
        tmp.obs[field_2].map(tmp.uns[f'{field_2}_colors_dict'])], axis=1)


# %%
# %%time

pairs = [(stages[i], stages[i + 1]) for i in range(len(stages) -1)]
pairs = [('P7', 'P15', )]

for stage_a, stage_b in pairs:
    comparison_dir = _utils.set_dir(results_dir.joinpath(f"epi_{stage_a}_{stage_b}"))
    
    adata_a = epi_split_adatas[stage_a]
    adata_b = epi_split_adatas[stage_b]
    
    if os.path.exists(comparison_dir.joinpath('comparator.npz')):
        cmp = comparator.Comparator.load_from_file(comparison_dir.joinpath('comparator.npz'), adata_a, adata_b)
    else:
        cmp = comparator.Comparator(adata_a, adata_a.obsm['usages'], adata_b, comparison_dir, 'sklearn', max_nmf_iter=1000, verbosity=1)
        
        print('decomposing')
        cmp.extract_geps_on_jointly_hvgs()
        cmp.decompose_b(repeats = 20)
    
    cmp.examine_adata_a_decomposition_on_jointly_hvgs()
    
    cmp.print_errors()
    cmp.examine_adata_b_decompositions()
    cmp.plot_decomposition_comparisons()
    
    cmp.calculate_fingerprints()
    
    print('running GSEA')
    cmp.run_gsea(gprofiler_kwargs=dict(organism='mmusculus', sources=['GO:BP', 'WP', 'REAC', 'KEGG', 'TF', 'HP']))

    cmp.save_to_file(comparison_dir.joinpath('comparator.npz'))

# %%
for stage_a, stage_b in pairs:
    comparison_dir = _utils.set_dir(results_dir.joinpath(f"epi_{stage_a}_{stage_b}"))
    
    adata_a = epi_split_adatas[stage_a]
    adata_b = epi_split_adatas[stage_b]
    
    cmp = comparator.Comparator.load_from_file(comparison_dir, adata_a, adata_b)
    break


# %%
var_names_gt_2 = sc.pp.filter_genes(epi_split_adatas['E15'].X, min_cells=2)[0]
np.sum(var_names_gt_2)

# %%
cmp
