import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tqdm import tqdm
import argparse

from data_loader import load_flow_data, load_unsw_nb15_data

MODELS_DIR = '../models/kitsune_ae'
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

MAX_AUTOENCODERS = 10
AE_HIDDEN = 64
EPOCHS = 50
BATCH_SIZE = 256

def create_feature_subsets(n_features, n_autoencoders):
    """Divide as features em n_autoencoders, criando subconjuntos aleatórios e disjuntos."""
    idx = np.arange(n_features)
    np.random.shuffle(idx)
    
    subsets = np.array_split(idx, n_autoencoders)
    return subsets

def build_small_ae(input_dim, hidden=AE_HIDDEN):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(hidden, activation='relu')(inp)
    x = layers.Dense(max(input_dim//2, 4), activation='relu')(x)
    x = layers.Dense(hidden, activation='relu')(x)
    out = layers.Dense(input_dim, activation='sigmoid')(x)
    m = models.Model(inp, out)
    m.compile(optimizer='adam', loss='mse')
    return m

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

    if x_train is None or x_test is None: return

    n_features = x_train.shape[1]
    subsets = create_feature_subsets(n_features, MAX_AUTOENCODERS)
    print(f"Criados {len(subsets)} sub-redes de features.")

    for i, inds in enumerate(tqdm(subsets, desc="Treinando Autoencoders")):
        model_path = os.path.join(MODELS_DIR, f'ae_subnet_{i}.h5')
        Xi_train = x_train[:, inds]
        ae = build_small_ae(len(inds))
        ae.fit(Xi_train, Xi_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        ae.save(model_path)

    print("\nIniciando a avaliação...")
    final_scores = np.zeros(x_test.shape[0])
    for i, inds in enumerate(tqdm(subsets, desc="Calculando scores de anomalia")):
        model_path = os.path.join(MODELS_DIR, f'ae_subnet_{i}.h5')
        ae = tf.keras.models.load_model(model_path, compile=False)
        Xi_test = x_test[:, inds]
        recon = ae.predict(Xi_test, verbose=0)
        mse = np.mean((Xi_test - recon)**2, axis=1)
        final_scores += mse

    y_test_numeric = np.array([label_mapping.get(str(label).strip(), -1) for label in y_test])
    benign_scores = final_scores[y_test_numeric == 0]

    if len(benign_scores) > 0:
        threshold = np.percentile(benign_scores, 99)
    else:
        threshold = np.percentile(final_scores, 99)
    print(f"\nThreshold dinâmico (FPR≈1%): {threshold:.6f}")

    y_pred_binary = (final_scores > threshold).astype(int)

    # Métricas
    fp = np.sum((y_pred_binary == 1) & (y_test_numeric == 0))
    tn = np.sum((y_pred_binary == 0) & (y_test_numeric == 0))
    tp = np.sum((y_pred_binary == 1) & (y_test_numeric != 0))
    fn = np.sum((y_pred_binary == 0) & (y_test_numeric != 0))
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tpr_geral = tp / (tp + fn) if (tp + fn) > 0 else 0

    print("\n========== RESULTADOS FINAIS (Kitsune-AE) ==========")
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
    parser = argparse.ArgumentParser(description='Treinar e Avaliar Kitsune-AE em diferentes datasets.')
    parser.add_argument('--dataset', type=str, default='flow', choices=['flow', 'unsw_nb15'],
                        help='O nome do dataset a ser usado (flow ou unsw_nb15).')
    args = parser.parse_args()
    train_and_evaluate(dataset_name=args.dataset)
