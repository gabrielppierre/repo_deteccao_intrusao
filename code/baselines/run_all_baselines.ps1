& {
    # Script para executar todos os baselines nos datasets CIC-IDS 2017 (flow) e UNSW-NB15.

    Write-Host "================================================="
    Write-Host "INICIANDO EXECUÇÃO DE TODOS OS BASELINES"
    Write-Host "=================================================`n"

    # --- DAGMM ---
    Write-Host "--- Executando DAGMM no dataset 'flow' (CIC-IDS 2017) ---"
    conda run -n repo python train_evaluate_dagmm.py --dataset flow
    Write-Host "`n--- Executando DAGMM no dataset 'unsw_nb15' (UNSW-NB15) ---"
    conda run -n repo python train_evaluate_dagmm.py --dataset unsw_nb15
    Write-Host "`n"

    # --- BiGAN / EGBAD ---
    Write-Host "--- Executando BiGAN/EGBAD no dataset 'flow' (CIC-IDS 2017) ---"
    conda run -n repo python train_evaluate_bigan.py --dataset flow --w 0.1 --d 1.0 --m fm
    Write-Host "`n--- Executando BiGAN/EGBAD no dataset 'unsw_nb15' (UNSW-NB15) ---"
    conda run -n repo python train_evaluate_bigan.py --dataset unsw_nb15 --w 0.1 --d 1.0 --m fm
    Write-Host "`n"

    # --- Kitsune-GMM ---
    Write-Host "--- Executando Kitsune-GMM no dataset 'flow' (CIC-IDS 2017) ---"
    conda run -n repo python train_evaluate_kitsune_gmm.py --dataset flow
    Write-Host "`n--- Executando Kitsune-GMM no dataset 'unsw_nb15' (UNSW-NB15) ---"
    conda run -n repo python train_evaluate_kitsune_gmm.py --dataset unsw_nb15
    Write-Host "`n"

    # --- Kitsune-AE ---
    Write-Host "--- Executando Kitsune-AE no dataset 'flow' (CIC-IDS 2017) ---"
    conda run -n repo python train_evaluate_kitsune_ae.py --dataset flow
    Write-Host "`n--- Executando Kitsune-AE no dataset 'unsw_nb15' (UNSW-NB15) ---"
    conda run -n repo python train_evaluate_kitsune_ae.py --dataset unsw_nb15
    Write-Host "`n"

    Write-Host "================================================="
    Write-Host "EXECUÇÃO DE TODOS OS BASELINES CONCLUÍDA"
    Write-Host "================================================="
} *>&1 | Tee-Object -FilePath "run_all_baselines_output.txt"
