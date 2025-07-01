import argparse
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from Blocking.src.tools.utils import load_pkl, save2pkl
from Matching.src.Conflict_Resolution import get_max_weight_matching_answers
from Matching.src.Feature_Extractor import get_feature_based_on_multi_fea_fusion, \
    get_feature_based_on_semantic
from Matching.src.Interaction_with_LLM import generate_prompts_and_parallel_interact_with_llm, \
    generate_prompts_and_interact_with_llm_limit_rate, generate_prompts_and_interact_with_llm_chatgpt
from Matching.src.Pair_Batching import generate_diverse_batches, generate_similar_batches, \
    generate_random_batches
from Matching.src.Pair_Clustering import hierarchical_noise_clustering, generate_cluster_mappings, \
    print_clustering_report, adaptable_clustering
from Matching.src.Performance_Measure import measure_performance
from Matching.src.tools.model import EncoderModel


def prompt_to_llm(batches, llm, ground_truth_path, save_data, city):
    # 1. 产生prompt并与llm交互
    if llm in ['DeepSeek-V2.5', 'DeepSeek-V3', 'DeepSeek-R1', 'Qwen3-32B', 'QwQ-32B', 'Qwen3-14B',
               'DeepSeek-R1-Distill-Qwen-14B', 'Qwen3-8B', 'DeepSeek-R1-0528-Qwen3-8B']:
        all_answers, all_confidences = generate_prompts_and_parallel_interact_with_llm(batches, 10, llm, city)
    elif llm in ['gemini-2.0-flash', 'gemini-2.5-flash-preview-04-17']:
        all_answers, all_confidences = generate_prompts_and_interact_with_llm_limit_rate(batches, llm, city)
    elif llm in ['gpt-4.1-mini', 'o4-mini']:
        all_answers, all_confidences = generate_prompts_and_interact_with_llm_chatgpt(batches, llm, city)

    # 2. 还原所有的候选实体对
    all_batches = []

    for batch in batches:
        all_batches.extend(batch)

    # 3. 量化初始匹配效果
    print('The performance of the ' + llm + ':')
    measure_performance(all_batches, all_answers, ground_truth_path)
    print()

    # 4. 冲突消解
    all_answers = get_max_weight_matching_answers(all_answers, all_confidences, all_batches)

    # 5. 量化冲突消解后的匹配效果
    print('After the Conflict Resolution:')
    measure_performance(all_batches, all_answers, ground_truth_path)

    save2pkl(all_answers, save_data)

    print('--------------------------------------------------------------')


if __name__ == '__main__':
    # 0. 参数处理
    parser = argparse.ArgumentParser()

    parser.add_argument('--city', type=str, default='nj', choices=['nj', 'hz', 'pit'],
                        help="The city to process. Defaults to 'nj'.")
    parser.add_argument('--feature_strategy', type=str, default='PROP_BASED', choices=['PROP_BASED', 'SEMANTIC_BASED'],
                        help="The feature extraction strategy. Defaults to 'PROP_BASED'.")
    parser.add_argument('--clustering_method', type=str, default='hdbscan',
                        choices=['hdbscan', 'kmeans', 'dbscan', 'agglomerative'],
                        help="The clustering method to use. Defaults to 'hdbscan'.")
    parser.add_argument('--batch_strategy', type=str, default='diverse', choices=['diverse', 'similar', 'random'],
                        help="The batch generation strategy. Defaults to 'diverse'.")
    parser.add_argument('--llm', type=str, default='DeepSeek-V3',
                        help="The name of the large language model to use.")

    args = parser.parse_args()

    city = args.city
    feature_strategy = args.feature_strategy
    clustering_method = args.clustering_method
    batch_strategy = args.batch_strategy
    llm = args.llm

    script_path = os.path.realpath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))

    if city == 'nj':
        ground_truth_path = os.path.join(project_root, 'data', 'nj', 'set_ground_truth_411.pkl')
        batches_save_path = os.path.join(project_root, 'Matching', 'batches_data', 'nj_batches.pkl')
    elif city == 'hz':
        ground_truth_path = os.path.join(project_root, 'data', 'hz', 'set_ground_truth_808.pkl')
        batches_save_path = os.path.join(project_root, 'Matching', 'batches_data', 'hz_batches.pkl')
    elif city == 'pit':
        ground_truth_path = os.path.join(project_root, 'data', 'pit', 'set_ground_truth_1237.pkl')
        batches_save_path = os.path.join(project_root, 'Matching', 'batches_data', 'pit_batches.pkl')
    else:
        raise ValueError(f"Unsupported city: '{city}'. Supported cities are 'nj', 'pit', 'hz'.")

    output_save_path = os.path.join(project_root, 'Matching', 'outputs', 'all_answers.pkl')

    full_blocking_outputs_path = os.path.join(project_root, 'Blocking', 'outputs')
    candidate_pairs_filename = f'{city}_candidate_pairs.pkl'
    candidate_pairs_full_address = os.path.join(full_blocking_outputs_path, candidate_pairs_filename)

    # 1. 提取空间实体分块结果
    candidate_pairs = load_pkl(candidate_pairs_full_address)
    candidate_pairs_list = list(candidate_pairs)

    # 2. 对候选实体对进行特征提取
    if feature_strategy == 'PROP_BASED':
        embeddings = get_feature_based_on_multi_fea_fusion(candidate_pairs_list, city)
    elif feature_strategy == 'SEMANTIC_BASED':
        encoder = EncoderModel()
        embeddings = get_feature_based_on_semantic(candidate_pairs_list, encoder)
    else:
        raise ValueError(f"Unsupported strategy: '{batch_strategy}'. Supported strategies are 'PROP_BASED', "
                         f"'SEMANTIC_BASED'")

    # 3. 候选实体对特征聚类 (分层hdbscan + 噪声knn投票)
    if clustering_method == 'hdbscan':
        final_labels = hierarchical_noise_clustering(
            embeddings
        )
    elif clustering_method in ['kmeans', 'dbscan', 'agglomerative']:
        final_labels = adaptable_clustering(
            embeddings, clustering_method
        )
    else:
        raise ValueError(f"Unsupported clustering method: '{clustering_method}'. Supported methods are 'hdbscan', "
                         f"'kmeans', 'dbscan' and 'agglomerative'")

    cluster_to_pairs, pair_to_cluster = generate_cluster_mappings(
        final_labels,
        candidate_pairs_list
    )

    print_clustering_report(final_labels)

    # 4. 候选实体对分组策略
    if batch_strategy == 'diverse':
        batches = generate_diverse_batches(cluster_to_pairs, batch_size=16)
    elif batch_strategy == 'similar':
        batches = generate_similar_batches(cluster_to_pairs, batch_size=16)
    elif batch_strategy == 'random':
        batches = generate_random_batches(cluster_to_pairs, batch_size=16)
    else:
        raise ValueError(f"Unsupported strategy: '{batch_strategy}'. Supported strategies are 'diverse', 'similar', "
                         f"'random'.")

    print('The number of all the batches: ' + str(len(batches)) + '\n')

    save2pkl(batches, batches_save_path)  # 保存batches，以便复现/对比

    # batches = load_pkl('')  # (把上述内容都注释掉)加载batches，以保证每次batches内容一样

    # 5. Matching with LLM
    prompt_to_llm(batches, llm, ground_truth_path, output_save_path, city)
