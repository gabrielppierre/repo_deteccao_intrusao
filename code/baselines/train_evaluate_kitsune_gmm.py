import os
import numpy as np
import joblib
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import pandas as pd
import argparse

from data_loader import load_flow_data, load_unsw_nb15_data

MODELS_DIR = '../models/kitsune_gmm'
RESULTS_DIR = '../results'
N_SUBNETS = 10

def get_feature_mapping(n_features, n_subnets):
    """Divide as features em subnets contíguas."""
    mapping = {}
    features_per_subnet = n_features // n_subnets
    for i in range(n_subnets):
        start = i * features_per_subnet
        end = (i + 1) * features_per_subnet if i < n_subnets - 1 else n_features
        mapping[i] = list(range(start, end))
    return mapping

def train_and_evaluate(dataset_name='flow'):
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"Carregando dataset: {dataset_name}")
    if dataset_name == 'flow':
        x_train, _, x_test, y_test = load_flow_data()
        label_mapping = {
            'BENIGN': 0, 'FTP-Patator': 1, 'SSH-Patator': 2, 'Slowloris': 3,
            'Slowhttptest': 4, 'Hulk': 5, 'GoldenEye': 6, 'Heartbleed': 7,
            'Web Attack  Brute Force': 8, 'Web Attack  XSS': 9, 'Web Attack  Sql Injection': 10,
            'Infiltration': 11, 'Bot': 12, 'PortScan': 13, 'DDoS': 14
        }
    elif dataset_name == 'unsw_nb15':
        x_train, _, x_test, y_test = load_unsw_nb15_data()
        label_mapping = {
            'Benign': 0, 'Analysis': 1, 'Backdoor': 2, 'DoS': 3, 'Exploits': 4,
            'Fuzzers': 5, 'Generic': 6, 'Reconnaissance': 7, 'Shellcode': 8, 'Worms': 9
        }
    else:
        raise ValueError("Dataset não suportado. Escolha 'flow' ou 'unsw_nb15'.")

    if x_train is None or x_test is None:
        print("Falha ao carregar dados. Abortando.")
        return

    print("Iniciando o treinamento do Kitsune-GMM...")
    n_features = x_train.shape[1]
    feature_mapping = get_feature_mapping(n_features, N_SUBNETS)

    for i, features_indices in tqdm(feature_mapping.items(), desc="Treinando GMMs para cada subnet"):
        subnet_data = x_train[:, features_indices]
        gmm = GaussianMixture(n_components=4, random_state=42, max_iter=200, n_init=5)
        gmm.fit(subnet_data)
        joblib.dump(gmm, os.path.join(MODELS_DIR, f'gmm_subnet_{i}.pkl'))
    print("Treinamento concluído.")

    print("\nIniciando a avaliação...")
    models = {i: joblib.load(os.path.join(MODELS_DIR, f'gmm_subnet_{i}.pkl')) for i in range(N_SUBNETS)}

    subnet_scores = []
    for i, features_indices in tqdm(feature_mapping.items(), desc="Calculando scores de anomalia"):
        subnet_data = x_test[:, features_indices]
        log_likelihood = models[i].score_samples(subnet_data)
        subnet_scores.append(-log_likelihood.reshape(-1, 1))
    
    final_scores = np.sqrt(np.mean(np.concatenate(subnet_scores, axis=1)**2, axis=1))

    y_test_numeric = np.array([label_mapping.get(str(label).strip(), -1) for label in y_test])
    benign_scores = final_scores[y_test_numeric == 0]
    
    if len(benign_scores) > 0:
        threshold = np.percentile(benign_scores, 99)
    else:
        threshold = np.percentile(final_scores, 99)
    print(f"\nThreshold dinâmico (FPR≈1%): {threshold:.6f}")

    y_pred_binary = (final_scores > threshold).astype(int)

    fp = np.sum((y_pred_binary == 1) & (y_test_numeric == 0))
    tn = np.sum((y_pred_binary == 0) & (y_test_numeric == 0))
    tp = np.sum((y_pred_binary == 1) & (y_test_numeric != 0))
    fn = np.sum((y_pred_binary == 0) & (y_test_numeric != 0))
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tpr_geral = tp / (tp + fn) if (tp + fn) > 0 else 0

    print("\n========== RESULTADOS FINAIS (Kitsune-GMM) ==========")
    print(f"FPR Geral: {fpr:.4f}")
    print(f"TPR Geral (Recall): {tpr_geral:.4f}")
    print("---------------------------------------------------------")
    print("TPR por Ataque:")

    inv_label_mapping = {v: k for k, v in label_mapping.items()}
    for label_idx in sorted(inv_label_mapping.keys()):
        if label_idx <= 0: continue
        attack_name = inv_label_mapping[label_idx]
        attack_indices = (y_test_numeric == label_idx)
        if np.sum(attack_indices) > 0:
            tp_attack = np.sum((y_pred_binary == 1) & (attack_indices))
            tpr_attack = tp_attack / np.sum(attack_indices)
            print(f"- {attack_name}: {tpr_attack:.4f}")
    print("=========================================================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Treinar e Avaliar Kitsune-GMM em diferentes datasets.')
    parser.add_argument('--dataset', type=str, default='flow', choices=['flow', 'unsw_nb15'],
                        help='O nome do dataset a ser usado (flow ou unsw_nb15).')
    args = parser.parse_args()
    train_and_evaluate(dataset_name=args.dataset)
