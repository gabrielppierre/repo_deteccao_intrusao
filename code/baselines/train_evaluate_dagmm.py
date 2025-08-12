import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tqdm import tqdm

import argparse
from data_loader import load_flow_data, load_unsw_nb15_data

MODELS_DIR = '../models/dagmm'
RESULTS_DIR = '../results'

class DAGMM(Model):
    def __init__(self, n_features, n_components=2, latent_dim=1, lambda1=0.1, lambda2=0.005):
        super(DAGMM, self).__init__()
        self.n_features = n_features
        self.n_components = n_components
        self.latent_dim = latent_dim
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.estimation_net = self._build_estimation_net()

    def _build_encoder(self):
        inputs = Input(shape=(self.n_features,))
        x = Dense(60, activation='tanh')(inputs)
        x = Dense(30, activation='tanh')(x)
        x = Dense(10, activation='tanh')(x)
        z = Dense(self.latent_dim, activation=None)(x)
        return Model(inputs, z, name='encoder')

    def _build_decoder(self):
        latent_inputs = Input(shape=(self.latent_dim,))
        x = Dense(10, activation='tanh')(latent_inputs)
        x = Dense(30, activation='tanh')(x)
        x = Dense(60, activation='tanh')(x)
        outputs = Dense(self.n_features, activation=None)(x)
        return Model(latent_inputs, outputs, name='decoder')

    def _build_estimation_net(self):
        inputs = Input(shape=(self.latent_dim + 2,))
        x = Dense(10, activation='tanh')(inputs)
        x = Dropout(0.5)(x)
        outputs = Dense(self.n_components, activation='softmax')(x)
        return Model(inputs, outputs, name='estimation_net')

    def call(self, inputs):
        z_c = self.encoder(inputs)
        x_hat = self.decoder(z_c)
        rec_cosine = tf.keras.losses.cosine_similarity(inputs, x_hat)
        rec_euclidean = tf.norm(inputs - x_hat, axis=1)

        z = tf.concat([z_c, tf.expand_dims(rec_euclidean, axis=1), tf.expand_dims(rec_cosine, axis=1)], axis=1)
        gamma = self.estimation_net(z)
        return z_c, x_hat, z, gamma

    def compute_loss(self, x, x_hat, gamma, z):
        reconstruction_error = tf.reduce_mean(tf.reduce_sum((x - x_hat)**2, axis=1))

        phi = tf.reduce_mean(gamma, axis=0)
        mu = tf.reduce_sum(tf.expand_dims(gamma, axis=-1) * tf.expand_dims(z, axis=1), axis=0) / tf.expand_dims(tf.reduce_sum(gamma, axis=0), axis=-1)
        z_minus_mu = tf.expand_dims(z, axis=1) - tf.expand_dims(mu, axis=0)
        
        z_expanded_1 = tf.expand_dims(z_minus_mu, -1)  # (batch, k, latent_dim, 1)
        z_expanded_2 = tf.expand_dims(z_minus_mu, -2)  # (batch, k, 1, latent_dim)
        outer_prod = tf.matmul(z_expanded_1, z_expanded_2) # (batch, k, latent_dim, latent_dim)

        gamma_expanded = tf.expand_dims(tf.expand_dims(gamma, -1), -1) # (batch, k, 1, 1)
        cov_mat_numerator = tf.reduce_sum(gamma_expanded * outer_prod, axis=0) # (k, latent_dim, latent_dim)

        gamma_sum_per_component = tf.reduce_sum(gamma, axis=0)
        gamma_sum_expanded = tf.expand_dims(tf.expand_dims(gamma_sum_per_component, -1), -1)

        cov_mat = cov_mat_numerator / (gamma_sum_expanded + 1e-6)
        
        energy = 0
        for k in range(self.n_components):
            cov_k = cov_mat[k] + (tf.eye(z.shape[1]) * 1e-6)
            z_minus_mu_k = z - mu[k]
            inv_cov_k = tf.linalg.inv(cov_k)
            mahalanobis_dist = tf.reduce_sum(tf.matmul(z_minus_mu_k, inv_cov_k) * z_minus_mu_k, axis=1)
            term1 = mahalanobis_dist
            term2 = tf.math.log(tf.linalg.det(cov_k) + 1e-6)
            energy += gamma[:, k] * (term1 + term2)

        gmm_energy = tf.reduce_mean(energy)
        
        cov_diag = tf.linalg.diag_part(cov_mat)
        cov_penalty = tf.reduce_sum(tf.divide(1, cov_diag + 1e-6))

        return reconstruction_error + self.lambda1 * gmm_energy + self.lambda2 * cov_penalty

    def get_energy(self, x):
        _, _, z, gamma = self.call(x)
        phi = tf.reduce_mean(gamma, axis=0)
        mu = tf.reduce_sum(tf.expand_dims(gamma, axis=-1) * tf.expand_dims(z, axis=1), axis=0) / tf.expand_dims(tf.reduce_sum(gamma, axis=0), axis=-1)
        z_minus_mu = tf.expand_dims(z, axis=1) - tf.expand_dims(mu, axis=0)
        z_expanded_1 = tf.expand_dims(z_minus_mu, -1)  # (batch, k, latent_dim, 1)
        z_expanded_2 = tf.expand_dims(z_minus_mu, -2)  # (batch, k, 1, latent_dim)
        
        outer_prod = tf.matmul(z_expanded_1, z_expanded_2) # (batch, k, latent_dim, latent_dim)

        gamma_expanded = tf.expand_dims(tf.expand_dims(gamma, -1), -1) # (batch, k, 1, 1)
        cov_mat_numerator = tf.reduce_sum(gamma_expanded * outer_prod, axis=0) # (k, latent_dim, latent_dim)

        gamma_sum_per_component = tf.reduce_sum(gamma, axis=0)
        gamma_sum_expanded = tf.expand_dims(tf.expand_dims(gamma_sum_per_component, -1), -1)

        cov_mat = cov_mat_numerator / (gamma_sum_expanded + 1e-6)
        
        energy = []
        for k in range(self.n_components):
            cov_k = cov_mat[k] + (tf.eye(z.shape[1]) * 1e-6)
            z_minus_mu_k = z - mu[k]
            inv_cov_k = tf.linalg.inv(cov_k)
            mahalanobis_dist = tf.reduce_sum(tf.matmul(z_minus_mu_k, inv_cov_k) * z_minus_mu_k, axis=1)
            term1 = mahalanobis_dist
            term2 = tf.math.log(tf.linalg.det(cov_k))
            energy.append(gamma[:, k] * (term1 + term2))
        return tf.reduce_sum(energy, axis=0)

def train_and_evaluate(dataset_name='flow'):
    print("Carregando dados para DAGMM...")
    print(f"Carregando dataset: {dataset_name}")
    if dataset_name == 'flow':
        x_train, y_train, x_test, y_test = load_flow_data()
        label_mapping = {
            'BENIGN': 0, 'FTP-Patator': 1, 'SSH-Patator': 2, 'Slowloris': 3,
            'Slowhttptest': 4, 'Hulk': 5, 'GoldenEye': 6, 'Heartbleed': 7,
            'Web Attack  Brute Force': 8, 'Web Attack  XSS': 9, 'Web Attack  Sql Injection': 10,
            'Infiltration': 11, 'Bot': 12, 'PortScan': 13, 'DDoS': 14
        }
    elif dataset_name == 'unsw_nb15':
        x_train, y_train, x_test, y_test = load_unsw_nb15_data()
        label_mapping = {
            'Benign': 0, 'Analysis': 1, 'Backdoor': 2, 'DoS': 3, 'Exploits': 4,
            'Fuzzers': 5, 'Generic': 6, 'Reconnaissance': 7, 'Shellcode': 8, 'Worms': 9
        }
    else:
        raise ValueError("Dataset não suportado. Escolha 'flow' ou 'unsw_nb15'.")

    if x_train is None or x_test is None:
        print("Falha ao carregar dados. Abortando.")
        return

    print("Mapeando rótulos para inteiros...")
    y_test_numeric = np.array([label_mapping.get(str(label).strip(), -1) for label in y_test])


    n_features = x_train.shape[1]
    dagmm = DAGMM(n_features=n_features, lambda1=0.1, lambda2=0.005)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    print("Iniciando treinamento do DAGMM...")
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(1024)
    for epoch in range(50):
        for step, x_batch in enumerate(tqdm(train_dataset, desc=f"Epoch {epoch+1}/10")):
            with tf.GradientTape() as tape:
                z_c, x_hat, z, gamma = dagmm(x_batch)
                loss = dagmm.compute_loss(x_batch, x_hat, gamma, z)
            grads = tape.gradient(loss, dagmm.trainable_weights)
            optimizer.apply_gradients(zip(grads, dagmm.trainable_weights))
        print(f"Epoch {epoch+1}, Loss: {loss.numpy():.4f}")

    print("\nCalculando scores de energia no conjunto de teste...")
    final_scores = dagmm.get_energy(x_test).numpy()

    benign_mask = (y_test_numeric == 0)
    benign_scores = final_scores[benign_mask]
    attack_scores = final_scores[~benign_mask]

    print("\n--- Análise dos Scores de Energia ---")
    if len(benign_scores) > 0:
        print(f"Benignos: Min={np.min(benign_scores):.2f}, Max={np.max(benign_scores):.2f}, Média={np.mean(benign_scores):.2f}")
        threshold = np.percentile(benign_scores, 99)
    else:
        print("Nenhum score benigno encontrado. Usando percentil 99 de todos os scores como fallback.")
        threshold = np.percentile(final_scores, 99)
    
    if len(attack_scores) > 0:
        print(f"Ataques:  Min={np.min(attack_scores):.2f}, Max={np.max(attack_scores):.2f}, Média={np.mean(attack_scores):.2f}")
    else:
        print("Nenhum score de ataque encontrado.")
    print("-------------------------------------\n")

    print(f"Threshold dinâmico (FPR≈1%): {threshold:.6f}\n")
    y_pred_binary = (final_scores > threshold).astype(int)

    fp = np.sum((y_pred_binary == 1) & (y_test_numeric == 0))
    tn = np.sum((y_pred_binary == 0) & (y_test_numeric == 0))
    tp = np.sum((y_pred_binary == 1) & (y_test_numeric != 0))
    fn = np.sum((y_pred_binary == 0) & (y_test_numeric != 0))
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tpr_geral = tp / (tp + fn) if (tp + fn) > 0 else 0

    print("\n========== RESULTADOS FINAIS (DAGMM) ==========")
    print(f"FPR Geral: {fpr:.4f}")
    print(f"TPR Geral (Recall): {tpr_geral:.4f}")
    print("--------------------------------------------------")
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
    print("=================================================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Treinar e Avaliar DAGMM em diferentes datasets.')
    parser.add_argument('--dataset', type=str, default='flow', choices=['flow', 'unsw_nb15'],
                        help='O nome do dataset a ser usado (flow ou unsw_nb15).')
    args = parser.parse_args()
    
    train_and_evaluate(dataset_name=args.dataset)
