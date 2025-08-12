import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

DATA_DIR = '../../data'

def load_packet_data(data_type='train'):
    """Carrega e pré-processa os dados baseados em pacotes (usado por RePO)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(script_dir)) # Sobe para 'repo'

    if data_type == 'train':
        path = os.path.join(base_dir, 'data', 'packet_based', 'monday')
        print("Carregando dados de treino (packet-based)...")
        x_train_chunks = []
        y_train_chunks = []
        try:
            # Carrega todos os arquivos part_XXX.npy, assumindo que a última coluna é o label
            for file_name in tqdm(sorted(os.listdir(path))):
                if file_name.startswith('part_') and file_name.endswith('.npy'):
                    data = np.load(os.path.join(path, file_name))
                    x_train_chunks.append(data[:, :-1]) # Features
                    y_train_chunks.append(data[:, -1])  # Labels
            
            if not x_train_chunks:
                print(f"Erro: Nenhum arquivo de dados ('part_*.npy') encontrado em {path}")
                return None, None, None, None

            x_train = np.concatenate(x_train_chunks, axis=0).astype(np.float32)
            y_train = np.concatenate(y_train_chunks, axis=0).astype(np.float32)
            
            # Retorna 4 valores para manter a consistência
            return x_train, y_train, None, None
        except FileNotFoundError:
            print(f"Erro: Diretório de dados de treino não encontrado em {path}")
            return None, None, None, None

    elif data_type == 'test':
        print("Carregando dados de teste (packet-based)...")
        # Carrega os arquivos de normalização do modelo RePO
        model_path = os.path.join(base_dir, 'models', 'packet_based_model')
        min_path = os.path.join(model_path, 'x_min.npy')
        max_path = os.path.join(model_path, 'x_max.npy')

        try:
            x_min = np.load(min_path)
            x_max = np.load(max_path)
        except FileNotFoundError:
            print(f"Erro: Arquivos de normalização não encontrados em {model_path}.")
            print("Execute o script de treino 'train_packet_based_model.py' primeiro.")
            return None, None, None, None

        # Carrega e normaliza os dados de teste
        test_days = ['tuesday', 'wednesday', 'thursday', 'friday']
        x_test_chunks, y_test_chunks = [], []
        for day in test_days:
            day_path = os.path.join(base_dir, 'data', 'packet_based', day)
            try:
                x_day = np.load(os.path.join(day_path, f'x_{day}.npy'))
                y_day = np.load(os.path.join(day_path, f'y_{day}.npy'))
                x_test_chunks.append(x_day)
                y_test_chunks.append(y_day)
            except FileNotFoundError:
                print(f"Aviso: Arquivos de teste para '{day}' não encontrados em {day_path}. Pulando...")

        if not x_test_chunks:
            print("Erro: Nenhum dado de teste foi encontrado.")
            return None, None, None, None

        x_test = np.concatenate(x_test_chunks, axis=0)
        y_test = np.concatenate(y_test_chunks, axis=0)
        
        epsilon = 1e-7
        # Garante que o divisor não seja zero
        denominator = x_max - x_min + epsilon
        x_test_normalized = (x_test - x_min) / denominator
        
        # Retorna 4 valores para manter a consistência
        return None, None, x_test_normalized, y_test

    # Retorno padrão caso data_type seja inválido
    return None, None, None, None
    # Constrói o caminho absoluto para os arquivos de normalização
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # O caminho é relativo a 'repo/code/baselines', então subimos dois níveis para 'repo'
    base_dir = os.path.dirname(os.path.dirname(script_dir))
    
    min_path = os.path.join(base_dir, 'models', 'packet_based_model', 'x_min.npy')
    max_path = os.path.join(base_dir, 'models', 'packet_based_model', 'x_max.npy')

    try:
        x_min = np.load(min_path)
        x_max = np.load(max_path)
    except FileNotFoundError:
        print("Erro: Arquivos de normalização (x_min.npy, x_max.npy) não encontrados.")
        print(f"Caminho procurado: {os.path.join(base_dir, 'models', 'kitsune_ae')}")
        print("Verifique se o modelo Kitsune foi treinado ou se os arquivos estão no lugar certo.")
        return None, None, None, None

    all_data_chunks = []
    for day in test_days:
        path = os.path.join(DATA_DIR, 'packet_based', day)
        print(f"Processando diretório de teste: {path}")
        if os.path.isdir(path):
            for file_name in tqdm(sorted(os.listdir(path))):
                if 'part' in file_name and file_name.endswith('.npy'):
                    chunk = np.load(os.path.join(path, file_name))
                    all_data_chunks.append(chunk)
        else:
            print(f"Aviso: Diretório não encontrado: {path}")

    if not all_data_chunks:
        print("Nenhum dado de teste encontrado.")
        return None, None

    # Concatena todos os pedaços de dados de teste
    test_data = np.concatenate(all_data_chunks, axis=0)

    # Separa features e labels
    x_test = test_data[:, :-1].astype(np.float32)
    y_test = test_data[:, -1].astype(int)

    # Normaliza o conjunto de teste completo
    x_test = (x_test - x_min) / (x_max - x_min + 1e-6)

    return None, None, x_test, y_test

def load_unsw_nb15_data():
    """Carrega, pré-processa e divide o dataset UNSW-NB15."""
    print("Iniciando carregamento e processamento do UNSW-NB15...")
    # Constrói o caminho absoluto para os dados a partir da localização deste script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, '..', '..') # Sobe dois níveis (de repo/code/baselines para repo/)
    
    file_path = os.path.join(base_dir, 'data', 'nusw-nb15', 'nusw-nb15.csv')
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {file_path}")
        return None, None, None, None

    # Remove colunas não-numéricas que não serão usadas como features diretas
    df.drop(['srcip', 'sport', 'dstip', 'dsport'], axis=1, inplace=True)

    # One-hot encode para features categóricas
    categorical_cols = ['proto', 'state', 'service']
    df = pd.get_dummies(df, columns=categorical_cols, dummy_na=True)

    # Separa features e labels
    y = df['attack_cat'].fillna('Benign') # Preenche NaNs na categoria de ataque
    y_binary = df['label']
    X = df.drop(['attack_cat', 'label'], axis=1)

    # Garante que todas as colunas de features sejam numéricas
    X = X.apply(pd.to_numeric, errors='coerce')
    X.fillna(0, inplace=True)

    # Divisão em treino e teste (80/20) estratificado pela label binária
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y_binary
    )

    # Limpeza de Inf e NaN (pós-split para evitar data leakage)
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_train.fillna(0, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.fillna(0, inplace=True)

    # Normalização
    scaler = MinMaxScaler()
    x_train_norm = scaler.fit_transform(X_train)
    x_test_norm = scaler.transform(X_test)
    
    print("Carregamento do UNSW-NB15 concluído.")
    return x_train_norm.astype(np.float32), y_train.values, x_test_norm.astype(np.float32), y_test.values

def load_flow_data():
    """Carrega e pré-processa os dados de fluxo, alinhado com o notebook de referência."""


    print("Iniciando carregamento e processamento de dados de fluxo (alinhado ao RePO)...")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', '..', 'data', 'flow_based')

    def _preprocess_df(df, scaler=None):
        """Função auxiliar para pré-processar um DataFrame."""
        df = df.copy()
        if 'Flow ID' in df.columns:
            df = df[df['Flow ID'] != 'Flow ID']

        feats = df.iloc[:, 8:]
        ds_port = df.iloc[:, 5]
        df_processed = pd.concat([ds_port, feats], axis=1)

        labels = df_processed.iloc[:, -1]
        features = df_processed.iloc[:, :-1]

        features = features.apply(pd.to_numeric, errors='coerce')
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        features.fillna(0, inplace=True)

        if scaler is None:
            scaler = MinMaxScaler()
            features_norm = scaler.fit_transform(features)
        else:
            features_norm = scaler.transform(features)
            
        return features_norm.astype(np.float32), labels.values, scaler

    # --- 1. Processar Dados de Treino ---
    train_file = os.path.join(data_dir, 'Monday-WH-generate-labeled.csv')
    print(f"Processando dados de treino: {train_file}")
    try:
        df_train = pd.read_csv(train_file)
    except FileNotFoundError:
        print(f"Erro Crítico: Arquivo de treino não encontrado em {train_file}")
        return None, None, None, None
    
    x_train, y_train, scaler = _preprocess_df(df_train)

    # --- 2. Processar Dados de Teste ---
    print("Processando dados de teste...")
    test_files = [
        'Tuesday-WH-generate-labeled.csv', 'Wednesday-WH-generate-labeled.csv',
        'Thursday-WH-generate-labeled.csv', 'Friday-WH-generate-labeled.csv'
    ]
    x_test_list, y_test_list = [], []
    for file_name in test_files:
        full_test_path = os.path.join(data_dir, file_name)
        try:
            df_test = pd.read_csv(full_test_path)
            x_test_part, y_test_part, _ = _preprocess_df(df_test, scaler=scaler)
            x_test_list.append(x_test_part)
            y_test_list.append(y_test_part)
        except FileNotFoundError:
            print(f"Aviso: Arquivo de teste ignorado: {full_test_path}")

    if not x_test_list:
        print("Erro: Nenhum arquivo de teste foi carregado.")
        return x_train, y_train, None, None

    x_test = np.concatenate(x_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    print("Carregamento de dados de fluxo concluído.")
    return x_train, y_train, x_test, y_test
