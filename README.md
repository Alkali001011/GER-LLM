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


## 📋 Requirements
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

## 🚀 Quick Start
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

### Reproducing the Main Results

To reproduce the main results for the **GER-LLM** pipeline, please follow the steps below. The process is divided into two main stages: (1) Generating candidate pairs via spatial blocking and (2) Performing entity matching with the LLM.

#### Step 1: AOI-aware Spatial Blocking

This stage processes the raw POI data to generate high-quality candidate pairs for matching. It involves classifying AOIs first, then running the blocking algorithm.

1.  **Run AOI Classification**. This step trains a model to understand the functional areas of interest for the given city data.
    * Navigate to the AOI classification directory:
        ```bash
        cd Blocking/src/AOI_classification
        ```
    * Execute the training script. The following command trains a model for the Nanjing (`nj`) dataset:
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
    * The trained models will be saved in the `Blocking/src/AOI_classification/saved_models/` directory.

2.  **Generate Candidate Pairs**. Using the classified AOIs, this step runs the quadtree splitting algorithm to produce the final candidate pairs file.
    * Navigate to the main blocking directory:
        ```bash
        cd Blocking/src
        ```
    * Run the main blocking script:
        ```bash
        python main_blocking.py --city nj
        ```
    * The generated candidate pairs (e.g., `nj_candidate_pairs.pkl`) will be saved in the `Blocking/outputs/` directory. This file is required for the next stage.

#### Step 2: Group-wise Matching with LLM

This is the final stage where the LLM assesses the candidate pairs generated in Step 1 to produce the final entity resolution results.

1.  **Run the Matching Pipeline**.
    * Navigate to the matching directory:
        ```bash
        cd Matching/src
        ```
    * Execute the main matching script. This command runs the entire pipeline including feature extraction, clustering, group-wise prompting, and conflict resolution:
        ```bash
        python main_matching.py \
          --city nj \
          --feature_strategy PROP_BASED \
          --clustering_method hdbscan \
          --batch_strategy diverse \
          --llm DeepSeek-V3
        ```

2.  **Check the Results**. The final matching results and logs will be saved in the `Matching/outputs/` directory. Please ensure this folder exists before running to avoid a 'file not found' error.


[//]: # (### AOI-aware Spatial Blocking)

[//]: # (#### AOI Classification)

[//]: # (To reproduce the results of Spatial Blocking, you can run the following command:)

[//]: # (1. )

[//]: # (```bash)

[//]: # (cd Blocking/src/AOI_classification)

[//]: # (```)

[//]: # (2. )

[//]: # (```bash)

[//]: # (python train_AOI_Classifier.py \)

[//]: # (  --city nj \)

[//]: # (  --fe bert \)

[//]: # (  --lr 3e-5 \)

[//]: # (  --alpha 2.0 \)

[//]: # (  --beta 1.0 \)

[//]: # (  --n_epochs 10 \)

[//]: # (  --batch_size 32 \)

[//]: # (  --max_len 128 \)

[//]: # (  --device cuda \)

[//]: # (  --save_model)

[//]: # (```)

[//]: # ()
[//]: # (#### AOI-aware Splitting of Quadtree)

[//]: # (1. )

[//]: # (```bash)

[//]: # (cd Blocking/src)

[//]: # (```)

[//]: # (2. )

[//]: # (```bash)

[//]: # (python main_blocking.py --city nj)

[//]: # (```)

[//]: # ()
[//]: # (### Group-wise Matching with LLM & Graph-based Conflict Resolution)

[//]: # (1. )

[//]: # (```bash)

[//]: # (cd Matching/src)

[//]: # (```)

[//]: # (2. )

[//]: # (```bash)

[//]: # (python main_matching.py \)

[//]: # (  --city nj \)

[//]: # (  --feature_strategy PROP_BASED \)

[//]: # (  --clustering_method hdbscan \)

[//]: # (  --batch_strategy diverse \)

[//]: # (  --llm DeepSeek-V3)

[//]: # (```)

[//]: # (To pre-train the GURP model, please following the steps below: )

[//]: # ()
[//]: # (1. Download the pre-trained kge embeddings from [UrbanKG_TransR_entity]&#40;<https://drive.google.com/file/d/1OHEU-XPutEmhOvP0To2VhVakNhxkbPdp/view?usp=sharing>&#41;. After downloading, move the dataset to the `data/nymhtkg/kge_pretrained_transR/TransR_UrbanKG_1` directory. )

[//]: # (- ```bash)

[//]: # (  mkdir -p data/nymhtkg/kge_pretrained_transR/TransR_UrbanKG_1)

[//]: # (  ```)

[//]: # (2. Download the prepared NYC Manhattan KG from [mht180_prepared]&#40;<https://drive.google.com/file/d/1KqQjyOSEXhcgJevWVRljhp2sC6nalKmd/view?usp=sharing>&#41;. After downloading, move the dataset to the `data/nymhtkg` directory.)

[//]: # (- ```bash)

[//]: # (  mkdir -p data/nymhtkg)

[//]: # (  ```)

[//]: # (3. You then can run the following command to pre-train the GURP model.)

[//]: # (- ```bash)

[//]: # (  python train_gurp.py)

[//]: # (  ```)

[//]: # (4. The training log will be saved in the `experiments/gurp_model` directory. Please ensure that this folder has been created following code structure before training, otherwise, you might encounter a 'file not found' error.)

## 💾 Dataset

- Geospatial entities in Nanjing:
  - `dp_poi_12176.csv`: entities collected from [Dianping](https://www.dianping.com/).
  - `mt_poi_828.csv`: entities collected from [Meituan](https://www.meituan.com/).
  - `aoi_180.csv`: aois extracted from the entities above.
  - `set_ground_truth_411.pkl`: the ground truth.
- Geospatial entities in Hangzhou:
  - `gd_poi_1982.csv`: entities collected from [Amap](https://www.amap.com/).
  - `dp_poi_2959.csv`: entities collected from [Dianping](https://www.dianping.com/).
  - `aoi_107.csv`: aois extracted from the entities above.
  - `set_ground_truth_808.pkl`: the ground truth.
- Geospatial entities in Pittsburgh:
  - `osm_poi_2383.csv`: entities collected from [OpenStreetMap](https://www.openstreetmap.org/).
  - `fsq_poi_2474.csv`: entities collected from [Foursquare](https://foursquare.com/).
  - `aoi_181.csv`: aois extracted from the entities above.
  - `set_ground_truth_1237.pkl`: the ground truth.
