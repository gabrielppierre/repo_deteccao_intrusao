import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

import argparse
from data_loader import load_flow_data, load_unsw_nb15_data

# --- Configurações ---
MODELS_DIR = '../models/bigan'
RESULTS_DIR = '../results'
LATENT_DIM = 32
EPOCHS = 20
BATCH_SIZE = 1024
W_DISC = 0.1
D_FM = 1.0

def build_generator(n_features):
    latent_input = Input(shape=(LATENT_DIM,))
    x = Dense(64, activation='relu')(latent_input)
    x = Dense(128, activation='relu')(x)
    output = Dense(n_features, activation='sigmoid')(x)
    return Model(latent_input, output, name='generator')

def build_encoder(n_features):
    data_input = Input(shape=(n_features,))
    x = Dense(128, activation='relu')(data_input)
    x = Dense(64, activation='relu')(x)
    output = Dense(LATENT_DIM, activation=None)(x)
    return Model(data_input, output, name='encoder')

def build_discriminator(n_features):
    data_input = Input(shape=(n_features,))
    latent_input = Input(shape=(LATENT_DIM,))
    
    x = tf.concat([data_input, latent_input], axis=-1)
    x = Dense(128)(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    fm_layer = Dense(64)(x)
    fm_layer = LeakyReLU(alpha=0.2)(fm_layer)
    
    output = Dense(1, activation='sigmoid')(fm_layer)
    
    return Model([data_input, latent_input], [output, fm_layer], name='discriminator')


def train_and_evaluate(dataset_name='flow', w_disc=W_DISC, d_fm=D_FM, loss_mode='fm'):
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("Carregando dados para BiGAN...")
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

    if x_train is None or x_test is None: return

    label_mapping = {
        'BENIGN': 0, 'FTP-Patator': 1, 'SSH-Patator': 2, 'Slowloris': 3,
        'Slowhttptest': 4, 'Hulk': 5, 'GoldenEye': 6, 'Heartbleed': 7,
        'Web Attack  Brute Force': 8, 'Web Attack  XSS': 9, 'Web Attack  Sql Injection': 10,
        'Infiltration': 11, 'Bot': 12, 'PortScan': 13, 'DDoS': 14
    }
    y_test_numeric = np.array([label_mapping.get(str(label).strip(), -1) for label in y_test])


    n_features = x_train.shape[1]

    # Construir modelos
    generator = build_generator(n_features)
    encoder = build_encoder(n_features)
    discriminator = build_discriminator(n_features)

    # Otimizadores
    gen_enc_optimizer = Adam(0.0002, 0.5)
    disc_optimizer = Adam(0.0002, 0.5)

    discriminator.compile(loss='binary_crossentropy', optimizer=disc_optimizer)

    discriminator.trainable = False
    real_input = Input(shape=(n_features,))
    z_input = Input(shape=(LATENT_DIM,))
    
    enc_output = encoder(real_input)
    gen_output = generator(z_input)
    
    disc_real_pred, fm_real = discriminator([real_input, enc_output])
    disc_fake_pred, fm_fake = discriminator([gen_output, z_input])

    fm_loss = tf.reduce_mean(tf.square(fm_real - fm_fake))

    gan_model = Model([real_input, z_input], [disc_real_pred, disc_fake_pred])

    # Perda do gerador/encoder (L_G)
    if loss_mode == 'fm':
        # L_G = L_D_G + d * L_fm
        gan_loss = 'binary_crossentropy'
        gan_model.add_loss(d_fm * fm_loss)
    else:
        gan_loss = ['binary_crossentropy', 'binary_crossentropy']

    gan_model.compile(loss=gan_loss, optimizer=gen_enc_optimizer)

    valid = np.ones((BATCH_SIZE, 1))
    fake = np.zeros((BATCH_SIZE, 1))

    print("Iniciando treinamento do BiGAN...")
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(buffer_size=x_train.shape[0]).batch(BATCH_SIZE, drop_remainder=True)

    for epoch in range(EPOCHS):
        for x_batch in tqdm(train_dataset, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            z = np.random.normal(size=(BATCH_SIZE, LATENT_DIM))
            x_fake = generator.predict(z, verbose=0)
            x_real_latent = encoder.predict(x_batch, verbose=0)

            d_loss_real = discriminator.train_on_batch([x_batch, x_real_latent], [valid, np.zeros((BATCH_SIZE, 64))])[0]
            d_loss_fake = discriminator.train_on_batch([x_fake, z], [fake, np.zeros((BATCH_SIZE, 64))])[0]
            d_loss = w_disc * d_loss_real + (1 - w_disc) * d_loss_fake

            z = np.random.normal(size=(BATCH_SIZE, LATENT_DIM))
            z = np.random.normal(size=(BATCH_SIZE, LATENT_DIM))
            if loss_mode == 'fm':
                g_loss = gan_model.train_on_batch([x_batch, z], valid)
            else:
                g_loss = gan_model.train_on_batch([x_batch, z], [fake, valid])

    print("\nCalculando scores de anomalia no conjunto de teste...")
    print("\nCalculando scores de anomalia no conjunto de teste...")

    _, fm_test = discriminator.predict(x_test, verbose=0)
    z_test = encoder.predict(x_test, verbose=0)
    x_gen = generator.predict(z_test, verbose=0)
    _, fm_gen = discriminator.predict([x_gen, z_test], verbose=0)
    
    final_scores = np.mean(np.square(fm_test - fm_gen), axis=1)

    # Avaliação
    benign_scores = final_scores[y_test_numeric == 0]
    if len(benign_scores) > 0:
        threshold = np.percentile(benign_scores, 99)
    else:
        threshold = np.percentile(final_scores, 99)
    print(f"Threshold dinâmico (FPR≈1%): {threshold:.6f}")

    y_pred_binary = (final_scores > threshold).astype(int)

    fp = np.sum((y_pred_binary == 1) & (y_test_numeric == 0))
    tn = np.sum((y_pred_binary == 0) & (y_test_numeric == 0))
    tp = np.sum((y_pred_binary == 1) & (y_test_numeric == 1))
    fn = np.sum((y_pred_binary == 0) & (y_test_numeric == 1))

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0 

    print("\n========== RESULTADOS FINAIS (BiGAN) ==========")
    print(f"FPR Geral: {fpr:.4f}")
    print(f"TPR Geral (Recall): {tpr:.4f}")
    print("--------------------------------------------------")
    print("TPR por Ataque:")

    inv_label_mapping = {v: k for k, v in label_mapping.items()}
    for label_idx in sorted(inv_label_mapping.keys()):
        if label_idx == 0: continue

        attack_name = inv_label_mapping[label_idx]
        attack_indices = (y_test_numeric == label_idx)
        
        if np.sum(attack_indices) > 0:
            tp_attack = np.sum((y_pred_binary == 1) & (attack_indices))
            tpr_attack = tp_attack / np.sum(attack_indices)
            print(f"- {attack_name}: {tpr_attack:.4f}")
    print("=================================================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Treinar e Avaliar BiGAN em diferentes datasets.')
    parser.add_argument('--dataset', type=str, default='flow', choices=['flow', 'unsw_nb15'],
                        help='O nome do dataset a ser usado (flow ou unsw_nb15).')
    parser.add_argument('--w', type=float, default=0.1, help='Peso da perda do discriminador para amostras reais.')
    parser.add_argument('--d', type=float, default=1.0, help='Peso da perda de feature matching.')
    parser.add_argument('--m', type=str, default='fm', choices=['fm', 'cross-e'], help='Modo de perda para o gerador (fm ou cross-e).')

    args = parser.parse_args()
    
    train_and_evaluate(dataset_name=args.dataset, w_disc=args.w, d_fm=args.d, loss_mode=args.m)
