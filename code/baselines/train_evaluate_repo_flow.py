import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tqdm import tqdm
import argparse
import joblib

from data_loader import load_flow_data, load_unsw_nb15_data

MODELS_DIR = '../models/repo_flow'
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

EPOCHS = 50
BATCH_SIZE = 1024

N_MASKS = 10
MASK_RATE = 0.5

def build_repo_ae(n_features):
    """Constrói o modelo Autoencoder para o RePO."""
    inp = layers.Input(shape=(n_features,))
    x = layers.Dense(int(n_features * 0.75), activation='relu')(inp)
    x = layers.Dense(int(n_features * 0.5), activation='relu')(x)
    encoded = layers.Dense(int(n_features * 0.25), activation='relu')(x)
    x = layers.Dense(int(n_features * 0.5), activation='relu')(encoded)
    x = layers.Dense(int(n_features * 0.75), activation='relu')(x)
    decoded = layers.Dense(n_features, activation='sigmoid')(x)
    
    autoencoder = models.Model(inp, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def evaluate_repo(model, x_test, n_masks=N_MASKS, mask_rate=MASK_RATE):
    """Calcula os scores de anomalia usando a metodologia RePO."""
    n_samples, n_features = x_test.shape
    all_scores = np.zeros(n_samples)

    for i in tqdm(range(n_samples), desc="Avaliando com RePO"):
        sample = x_test[i]
        reconstruction_errors = []
        for _ in range(n_masks):
            mask = np.random.binomial(1, mask_rate, size=n_features).astype(bool)
            masked_sample = sample.copy()
            masked_sample[mask] = 0
            
            reconstructed = model.predict(np.array([masked_sample]), verbose=0)[0]
            error = np.mean((reconstructed[mask] - sample[mask])**2)
            reconstruction_errors.append(error)
        
        all_scores[i] = np.mean(reconstruction_errors)
        
    return all_scores

def train_and_evaluate(dataset_name='flow'):
    model_save_dir = os.path.join(MODELS_DIR, dataset_name)
    os.makedirs(model_save_dir, exist_ok=True)

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
    print(f"Número de features: {n_features}")
    autoencoder = build_repo_ae(n_features)
    
    print("Iniciando treinamento do Autoencoder RePO...")
    autoencoder.fit(x_train, x_train, 
                    epochs=EPOCHS, 
                    batch_size=BATCH_SIZE, 
                    shuffle=True, 
                    validation_split=0.1, 
                    verbose=1)
    
    model_path = os.path.join(model_save_dir, 'repo_ae.h5')
    autoencoder.save(model_path)
    print(f"Modelo salvo em: {model_path}")

    print("\nIniciando avaliação com a metodologia RePO...")
    final_scores = evaluate_repo(autoencoder, x_test)

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

    print("\n========== RESULTADOS FINAIS (RePO-Flow) ==========")
    print(f"Dataset: {dataset_name}")
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
    parser = argparse.ArgumentParser(description='Treinar e Avaliar o modelo RePO em diferentes datasets.')
    parser.add_argument('--dataset', type=str, default='flow', choices=['flow', 'unsw_nb15'],
                        help='O nome do dataset a ser usado (flow ou unsw_nb15).')
    args = parser.parse_args()
    train_and_evaluate(dataset_name=args.dataset)
