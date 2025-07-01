<!-- PROJECT LOGO -->
<br />
<div align="center">
    <img height="100" src="figure\logo.png?sanitize=true" />
</div>


<h2 align="center">🗺️ GER-LLM: Efficient and Effective Geospatial Entity Resolution with Large Language Model </h2>

<div align="center">

| **[Overview](#overview)** | **[Requirements](#requirements)** | **[Quick Start](#quick-start)** | **[Dataset](#Dataset)** |

</div>

🗺️ **GER-LLM** is a novel framework for **G**eospatial **E**ntity **R**esolution using **L**arge **L**anguage **M**odels. The core challenge in Geospatial Entity Resolution (GER) is to accurately identify whether different textual descriptions or records refer to the same real-world geographical entity. To tackle this, GER-LLM leverages the powerful semantic understanding and reasoning capabilities of LLMs. Our framework first generates high-quality candidate pairs from geospatial datasets. Then, it employs a sophisticated LLM-based featurizer to extract rich semantic, spatial, and contextual cues from entity descriptions. Finally, a lightweight yet effective matching model determines the final correspondence. Extensive experiments on benchmark datasets demonstrate that GER-LLM achieves state-of-the-art performance in both efficiency and effectiveness. This repository hosts the official source code for **GER-LLM**.

## Overview

The overall framework of GER-LLM is illustrated below:

<img src='figure\framework.png' alt="framework" >

The GER-LLM pipeline executes in three primary stages:
1. Perform AOI-aware spatial blocking to generate candidate pairs.
2. Use group-wise matching to jointly assess candidate groups with an LLM.
3. Apply a graph-based mechanism to resolve conflicts and ensure global consistency.

<!-- [//]: # (More details to come after accepted.) -->


## Requirements
- python==3.8.18
- aiohttp==3.10.11
- hdbscan==0.8.39
- numpy==1.24.4
- openai==1.93.0
- pandas==2.0.3
- python-dotenv==1.1.1
- python_Levenshtein==0.27.1
- PyYAML==6.0.2
- scikit_learn==1.3.2
- scipy==1.9.3
- torch==2.1.1
- transformers==4.46.3
- wget==3.2

## Quick Start
### Code Structure
```bash
+---GER-LLM
|   +---Blocking
|   |   +---outputs
|   |   |       hz_candidate_pairs.pkl
|   |   |       nj_candidate_pairs.pkl
|   |   |       pit_candidate_pairs.pkl
|   |   +---processed_data
|   |   |   +---hz
|   |   |   |       hz_poi_id2se.pkl
|   |   |   +---nj
|   |   |   |       nj_poi_id2se.pkl
|   |   |   \---pit
|   |   |           pit_poi_id2se.pkl
|   |   \---src
|   |       +---AOI_classification
|   |       |       config.py
|   |       |       functions.py
|   |       |       model_functions.py
|   |       |       models.py
|   |       |       refinement.py
|   |       |       saved_models
|   |       |       train_AOI_Classifier.py
|   |       |       train.py
|   |       |   main_blocking.py
|   |       \---tools
|   |               dbscan.py
|   |               Quadtree.py
|   |               SE.py
|   |               utils.py
|   +---data
|   |   +---hz
|   |   |   |   aoi_107.csv
|   |   |   |   dp_poi_2959.csv
|   |   |   |   gd_poi_1982.csv
|   |   |   |   set_ground_truth_808.pkl
|   |   |   \---hz
|   |   |           test.csv
|   |   |           train.csv
|   |   |           valid.csv
|   |   +---nj
|   |   |   |   aoi_180.csv
|   |   |   |   dp_poi_12176.csv
|   |   |   |   mt_poi_828.csv
|   |   |   |   set_ground_truth_411.pkl
|   |   |   \---nj
|   |   |           test.csv
|   |   |           train.csv
|   |   |           valid.csv
|   |   \---pit
|   |       |   aoi_181.csv
|   |       |   fsq_poi_2474.csv
|   |       |   osm_poi_2383.csv
|   |       |   set_ground_truth_1237.pkl
|   |       \---pit
|   |               test.csv
|   |               train.csv
|   |               valid.csv
|   +---figure
|   |       framework.png
|   |       logo.png
|   +---Matching
|   |   +---batches_data
|   |   |       hz_batches.pkl
|   |   |       nj_batches.pkl
|   |   |       pit_batches.pkl
|   |   +---outputs
|   |   \---src
|   |       |   Batch_Prompting.py
|   |       |   Conflict_Resolution.py
|   |       |   Feature_Extractor.py
|   |       |   Interaction_with_LLM.py
|   |       |   main_matching.py
|   |       |   Pair_Batching.py
|   |       |   Pair_Clustering.py
|   |       |   Performance_Measure.py
|   |       \---tools
|   |               model.py
|   |               utils.py
|   |
|   |   README.md
|   |   requirements.txt
```

### AOI-aware Spatial Blocking
#### AOI Classification
To reproduce the results of Spatial Blocking, you can run the following command:
```bash
cd Blocking/src/AOI_classification
```
```bash
python train_AOI_Classifier.py \
  --city nj \
  --fe bert \
  --lr 3e-5 \
  --alpha 2.0 \
  --beta 1.0 \
  --n_epochs 10 \
  --batch_size 32 \
  --max_len 128 \
  --device cuda \
  --save_model
```
The test log will be saved in the `experiments/gurp_prompt` directory. Please ensure that this folder has been created following code structure before testing, otherwise, you might encounter a 'file not found' error.
#### AOI-aware Splitting of Quadtree
```bash
cd Blocking/src
```
```bash
python main_blocking.py --city nj
```

### Group-wise Matching with LLM & Graph-based Conflict Resolution
```bash
cd Matching/src
```
```bash
python main_matching.py \
  --city nj \
  --feature_strategy PROP_BASED \
  --clustering_method hdbscan \
  --batch_strategy diverse \
  --llm DeepSeek-V3
```
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

## Dataset

- Geospatial entities in Nanjing:
  - `dp_poi_12176.csv`: The urban region graph in Manhattan of NYC.
  - `mt_poi_828.csv`: The urban region graph in Manhattan of NYC.
  - `aoi_180.csv`: The urban region graph in Manhattan of NYC.
  - `set_ground_truth_411.pkl`: The urban region graph in Manhattan of NYC.
- Geospatial entities in Hangzhou:
  - `gd_poi_1982.csv`: 
  - `dp_poi_2959.csv`: 
  - `aoi_107.csv`: 
  - `set_ground_truth_808.pkl`: 
- Geospatial entities in Pittsburgh:
  - `osm_poi_2383.csv`: 
  - `fsq_poi_2474.csv`: 
  - `aoi_181.csv`: 
  - `set_ground_truth_1237.pkl`: 
