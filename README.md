<!-- PROJECT LOGO -->
<br />
<div align="center">
    <img height="100" src="figure\logo.png?sanitize=true" />
</div>


<h2 align="center">🌇 Urban Region Pre-training and Prompting: A Graph-based Approach </h2>

<div align="center">

| **[Overview](#overview)** | **[Requirements](#requirements)** | **[Quick Start](#quick-start)** | **[Dataset](#Dataset)** |

</div>

🌇**GURPP** is a **G**raph-based **U**rban **R**egion **P**re-training and **P**rompting framework for region representation learning. Specifically, we first construct an **urban region graph** that integrates detailed spatial entity data for more effective urban region representation. Then, we develop a **subgraph-centric urban region pre-training model** to capture the heterogeneous and transferable patterns of interactions among entities. To further enhance the adaptability of these embeddings to different tasks, we design **two graph-based prompting methods** to incorporate explicit/hidden task knowledge.  Extensive experiments on various urban region prediction tasks and different cities demonstrate the superior performance of our GURPP framework. This repository hosts the code of **GURPP**.


## Overview

The Overview of GURPP is shown as follows:

<img src='figure\img_1.png' alt="framework" >

1. Create urban region graph that integrates specific urban entities.
2. Extract subgraph of each region according to the graph pattern.
3. Use multi-view learning to learn the subgraph representations with GURP model.
4. Adapt the region embeddings with manually-designed prompt method and task-learnable prompt learning method.


<!-- [//]: # (More details to come after accepted.) -->


## Requirements
- python==3.8.8
- pytorch==1.13.1
- numpy==1.23.5
- dgl==2.2.1
- scikit-learn==1.2.0
- pandas==1.5.2

## Quick Start
### Code Structure
```bash
+---GURPP
|   |   downstream_task.py
|   |   emb.npy
|   |   gurpp_args.py
|   |   load_graph_data.py
|   |   loss.py
|   |   README.md
|   |   test_gurp.py
|   |   train_gurp.py
|   |   train_learnable_prompt.py
|   |
|   +---data
|   |   +---nymhtkg
|   |   |   |   flow_in_per_hour.npy
|   |   |   |   flow_out_per_hour.npy
|   |   |   |   mobility_distribution.npy
|   |   |   |   region_si_img.npy
|   |   |   |
|   |   |   \---kge_pretrained_transR
|   |   |       \---TransR_UrbanKG_1
|   |   \---task
|   |           carbon_counts.npy
|   |           check_counts.npy
|   |           crime_counts.npy
|   |           income_counts.npy
|   |
|   +---experiments
|   |   \---gurp_model
|   |   \---gurp_prompt
|   |
|   +---figure
|   \---model
|           gurp.py
|           hgt.py
|           prompt.py
```

### Reproduce
To reproduce the GURP results, you can run the following command:
```bash
python test_gurp.py
```
The test log will be saved in the `experiments/gurp_prompt` directory. Please ensure that this folder has been created following code structure before testing, otherwise, you might encounter a 'file not found' error.

### Pre-train GURP
To pre-train the GURP model, please following the steps below: 

1. Download the pre-trained kge embeddings from [UrbanKG_TransR_entity](<https://drive.google.com/file/d/1OHEU-XPutEmhOvP0To2VhVakNhxkbPdp/view?usp=sharing>). After downloading, move the dataset to the `data/nymhtkg/kge_pretrained_transR/TransR_UrbanKG_1` directory. 
- ```bash
  mkdir -p data/nymhtkg/kge_pretrained_transR/TransR_UrbanKG_1
  ```
2. Download the prepared NYC Manhattan KG from [mht180_prepared](<https://drive.google.com/file/d/1KqQjyOSEXhcgJevWVRljhp2sC6nalKmd/view?usp=sharing>). After downloading, move the dataset to the `data/nymhtkg` directory.
- ```bash
  mkdir -p data/nymhtkg
  ```
3. You then can run the following command to pre-train the GURP model.
- ```bash
  python train_gurp.py
  ```
4. The training log will be saved in the `experiments/gurp_model` directory. Please ensure that this folder has been created following code structure before training, otherwise, you might encounter a 'file not found' error.

### Prompt
#### Task-learnable Prompt

To train task-learnable prompt model, you can run the following command.
```bash
python train_learnable_prompt.py
```
And the training log will be saved in the `experiments/gurp_prompt` directory. Please ensure that this folder has been created following code structure before training, otherwise, you might encounter a 'file not found' error.

#### Manually-designed Prompt

Manually-designed prompt can be designed by following steps.

1. Set the construct parameter `if_test` of GURPData as `true`.

2. Modify the method `get_region_sub_test_all()` in `load_graph_data.py` and then input them to the pretrained model to infer the results.
```python
# fanout configuration to control the inclusion or exclusion of specific data types
# 0: exclude (ignore) the information
# -1: include the information
fanout = {'JCateOf': 0, 'BrandOf': 0, 'Cate1Of': 0, 'HasJunc': -1, 'HasPoi': -1, 'HasRoad': -1,
            'NearBy': -1, 'RCateOf': 0}
sub_graph = dgl.sampling.sample_neighbors(self.hg, {'region': i}, fanout,
                                            edge_dir='out', copy_ndata=True, copy_edata=True)

sub_poi_nodes = sub_graph.edges(etype='HasPoi')[1]
sub_road_nodes = sub_graph.edges(etype='HasRoad')[1]
sub_junc_nodes = sub_graph.edges(etype='HasJunc')[1]
sub_region_nodes = sub_graph.edges(etype='NearBy')[1]

seed = 0
random.seed(seed)
# Randomly sample 90% of the POI nodes, road nodes and junction nodes
sub_poi_nodes = random.sample(sub_poi_nodes.tolist(), int(len(sub_poi_nodes) * 0.9))
sub_road_nodes = random.sample(sub_road_nodes.tolist(), int(len(sub_road_nodes) * 0.9))
sub_junc_nodes = random.sample(sub_junc_nodes.tolist(), int(len(sub_junc_nodes) * 0.9))

sub_poi_nodes = torch.tensor(sub_poi_nodes, dtype=torch.long).to(self.graph_device)
sub_road_nodes = torch.tensor(sub_road_nodes, dtype=torch.long).to(self.graph_device)
sub_junc_nodes = torch.tensor(sub_junc_nodes, dtype=torch.long).to(self.graph_device)

# Sample neighbors for POI brand, POI category, junction category and road category
# 0: exclude (ignore) the information
# -1: include the information
poi_brand_graph = dgl.sampling.sample_neighbors(self.hg, {'poi': sub_poi_nodes},
                                                {'JCateOf': 0, 'BrandOf': -1, 'Cate1Of': 0, 'HasJunc': 0,
                                                    'HasPoi': 0, 'HasRoad': 0, 'NearBy': 0, 'RCateOf': 0},
                                                edge_dir='out', copy_ndata=True, copy_edata=True)
poi_cate1_graph = dgl.sampling.sample_neighbors(self.hg, {'poi': sub_poi_nodes},
                                                {'JCateOf': 0, 'BrandOf': 0, 'Cate1Of': -1, 'HasJunc': 0,
                                                    'HasPoi': 0, 'HasRoad': 0, 'NearBy': 0, 'RCateOf': 0},
                                                edge_dir='out', copy_ndata=True, copy_edata=True)
junc_cate_graph = dgl.sampling.sample_neighbors(self.hg, {'junction': sub_junc_nodes},
                                                {'JCateOf': -1, 'BrandOf': 0, 'Cate1Of': 0, 'HasJunc': 0,
                                                    'HasPoi': 0, 'HasRoad': 0, 'NearBy': 0, 'RCateOf': 0},
                                                edge_dir='out', copy_ndata=True, copy_edata=True)
road_cate_graph = dgl.sampling.sample_neighbors(self.hg, {'road': sub_road_nodes},
                                                {'JCateOf': 0, 'BrandOf': 0, 'Cate1Of': 0, 'HasJunc': 0,
                                                    'HasPoi': 0, 'HasRoad': 0, 'NearBy': 0, 'RCateOf': -1},
                                                edge_dir='out', copy_ndata=True, copy_edata=True)
```

## Dataset

- Urban Region Graph in Manhattan of NYC (in the format of triples):
  - `mht180_prepared.csv`: The urban region graph in Manhattan of NYC.
- Pretrained nymht KGE embeddings for urban region entities:
  - `TransR_UrbanKG_1`: The pretrained nymht KGE embeddings for urban region entities.
