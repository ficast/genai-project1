# Notes for Report

---

## Etapas de Desenvolvimento

### Phase 1 — Exploração inicial com subset 10k (`phase1_subset_baseline.ipynb`)

Primeiro contacto com o dataset e com as arquitecturas. Todos os modelos foram treinados no **subset de 10k imagens** (20% do dataset) para validar pipelines e obter FIDs de referência rápidos.

| Modelo | FID (10k subset) | Nota |
|---|---|---|
| VAE | 241.61 | Reconstrução difusa, KL collapse parcial |
| DCGAN | 91.51 | Modo collapse parcial já visível |
| cGAN | 170.13 | Conditioning por classe não ajudou |
| DDPM | 146.54 | Sub-amostrado com sampler DDIM (200 steps) |
| DDIM | 97.87 | Melhor modelo no subset |

No final desta fase, o DDIM foi treinado pela primeira vez no **dataset completo 50k**, obtendo FID=44.99 — resultado que se tornaria o baseline de referência.

**Conclusão da fase**: DDIM claramente superior às restantes arquitecturas no subset. Foco das próximas fases no DDIM com 50k.

---

### Phase 2a — Optimizações DDIM: cosine schedule e EMA (`phase2a_opt_ema.ipynb`)

Duas experiências sobre o baseline DDIM 50k (FID=45.01), mantendo tudo o resto igual:

**DDIM Optimized — Cosine schedule + 300 epochs**
- Schedule linear (β₁=1e-4 → β_T=2e-2) substituído por cosine schedule (Nichol & Dhariwal 2021)
- Treino prolongado de 100 → 300 epochs
- **Resultado**: FID=57.75 — piorou (+12.74). O cosine schedule não beneficiou imagens 32×32 neste regime; 300 epochs introduziram overfitting.

**DDIM + EMA — 200 epochs, schedule linear**
- Exponential Moving Average dos pesos (decay=0.9999), usado só na inferência
- **Resultado**: FID=88.59 (avaliado com apenas 1000 amostras — menos fiável). Piorou significativamente. EMA não convergiu ou implementação teve problema.

**Conclusão da fase**: nenhuma das duas técnicas melhorou o baseline. O DDIM simples com 100 epochs e schedule linear manteve-se imbatível.

---

### Phase 2b — Optuna, Optuna Config e Attention (`phase2b_optuna_attn.ipynb`)

Três novas abordagens, ainda sobre o baseline DDIM 50k:

**DDPM Full (50k, 100 epochs)**
- Treino do DDPM estocástico completo com 50k imagens
- **Resultado**: FID=69.46 — significativamente pior que DDIM (45.01). Confirma que a amostragem determinística DDIM é superior.

**Optuna Hyperparameter Search**
- Pesquisa automática de hiperparâmetros (lr, hidden_dims, time_emb_dim, batch_size) com Optuna
- Melhores parâmetros aplicados ao run `ddim_full_optcfg_ep100_seed42`
- **Resultado**: FID=71.81 (avaliado com 1000 amostras — menos fiável). Piorou vs baseline.

**DDIM + Attention (50k, 100 epochs)**
- Mecanismo de atenção adicionado à UNet no bottleneck
- **Resultado**: FID=63.68 — piorou (+18.67). Atenção introduz complexidade que não é justificada pela escala do dataset (32×32, 50k).

**Conclusão da fase**: todas as tentativas de melhoria da arquitectura ou dos hiperparâmetros falharam. O baseline DDIM simples continua a ser o melhor modelo treinado.

---

### Full Training — Notebook de produção (`full_training.ipynb`)

Notebook limpo construído a partir das fases anteriores. Inclui apenas treino com 50k imagens, sem cGAN nem subset 10k. Arquitecturas cobertas: VAE, DCGAN, DDPM, DDIM. Resultados de referência:

| Modelo | FID | KID×10³ |
|---|---|---|
| VAE | 195.70 | 185.98 |
| DCGAN | 202.48 | 91.79 |
| DDPM | 69.46 | 47.82 |
| DDIM | 45.01 | 24.96 |

---

### Experiências de Optimização — Notebooks v2/v3/v4

Após consolidação do `full_training.ipynb`, iniciou-se uma fase de optimização sistemática com notebooks dedicados, cada um com **uma única variável alterada** face ao baseline.

#### DCGAN — Combate ao mode collapse

**DCGAN v2** (`dcgan_v2.ipynb`): n_critic=2, label smoothing=0.9, sem grad clip em D
- Mode collapse persistiu. BCE loss é o problema raiz — quando D satura, gradiente chega ao G quase a zero.

**DCGAN v3** (`dcgan_v3.ipynb`): WGAN-GP — Wasserstein loss + gradient penalty (λ=10), InstanceNorm, n_critic=5, lr=1e-4, betas=(0,0.9)
- FID=**102.29** — melhoria de ~50% face ao baseline (202.48). Melhor resultado GAN obtido.

#### DDIM — Optimizações de treino (todas sem melhoria)

**DDIM v2** (`ddim_v2.ipynb`): Adam → AdamW(weight_decay=1e-4) | FID=45.15 (+0.14)

**DDIM v3** (`ddim_v3.ipynb`): hidden_dims=[64,128,256,256] | FID=65.50 (+20.49)
- Loss plateou ao mesmo nível de v1 (~0.033). Capacidade extra não convergiu em 100 epochs.

#### DDIM — Optimização de inferência (melhor resultado)

**DDIM v4** (`ddim_v4.ipynb`): sweep de inference steps com checkpoint v1 (sem retreino)

| Steps | FID | KID×10³ |
|---|---|---|
| 200 (baseline) | 45.01 | 24.96 |
| 225 | (a medir) | — |
| **250** | **43.18** | **23.70** |
| 275 | (a medir) | — |
| 300 | (a medir) | — |
| 500 | 49.64 | 31.70 |

**250 steps é o melhor resultado global até ao momento (FID=43.18).** O sweet spot existe porque abaixo do óptimo a ODE fica sub-amostrada, acima o erro de discretização acumula-se.

---

## DCGAN — Mode Collapse

### O que aconteceu

O DCGAN treinado na versão baseline (`dcgan_full_ep100_seed42`) apresentou **mode collapse**: as 36 amostras geradas mostram composições repetidas — o Generator aprendeu a mapear vectores `z` distintos para outputs semelhantes. FID=202.5 confirma a baixa diversidade.

### Causa

Mode collapse é uma instabilidade de treino em GANs onde:

1. O **Generator** descobre um subconjunto de imagens que sistematicamente "engana" o Discriminator.
2. O **Discriminator** não consegue recuperar a tempo (treino 1:1, gradiente clipado em ambos).
3. O Generator converge para produzir apenas esse subconjunto — mesmo com `z` completamente diferentes.

Factores que agravaram nesta configuração:
- `n_critic = 1` — D não tem iterações suficientes para aprender antes do G atacar
- `clip_grad_norm(D, max_norm=1.0)` — limita a capacidade do D de recuperar quando colapsado
- Sem label smoothing — D torna-se demasiado confiante com labels binárias duras, facilitando o G a explorar essa certeza

### DCGAN v2 — n_critic=2, label smoothing, sem clip D

Notebook: `dcgan_v2.ipynb` | Run: `dcgan_v2_ep100_seed42`

| Parâmetro | v1 (baseline) | v2 |
|---|---|---|
| `n_critic` | 1 | **2** |
| Grad clip D | `max_norm=1.0` | **removido** |
| Grad clip G | `max_norm=1.0` | `max_norm=1.0` |
| Label real | 1.0 | **0.9** (label smoothing) |

**Resultado**: mode collapse persistiu. As imagens apresentam maior diversidade superficial mas ainda colapso evidente. FID não medido (treino rejeitado).

**Conclusão**: BCE loss em si é o problema raiz — quando D satura, o gradiente que chega ao G é quase zero independentemente das outras correcções.

### DCGAN v3 — WGAN-GP

Notebook: `dcgan_v3.ipynb` | Run: `dcgan_v3_ep100_seed42` | FID=**102.29** (−50% vs baseline)

| Parâmetro | v2 | v3 |
|---|---|---|
| Loss | BCE | **Wasserstein** |
| Normalização Critic | BatchNorm | **InstanceNorm** |
| `n_critic` | 2 | **5** |
| `lambda_gp` | — | **10** |
| `lr` | 2e-4 | **1e-4** |
| `betas` | (0.5, 0.999) | **(0, 0.9)** |
| Label smoothing | 0.9 | **removido** |

**Resultado**: FID=102.29, KID=84.25×10⁻³. Redução de ~50% no FID face ao baseline (202.48). O WGAN-GP eliminou o mode collapse ao substituir a BCE instável pela distância Wasserstein, que fornece gradientes informativos ao Generator mesmo quando o Critic é forte. É o melhor resultado GAN obtido no projecto, embora ainda significativamente acima do DDIM (FID=43.18).

---

## DDIM — Experiências de Optimização

### Baseline

`ddim_full_ep100_seed42` — FID=**45.01** | Adam, lr=2e-4, hidden_dims=[64,128,256], 200 DDIM steps

### DDIM v2 — AdamW (sem melhoria)

Notebook: `ddim_v2.ipynb` | Run: `ddim_v2_adamw_ep100_seed42` | FID=**45.15**

Única mudança: Adam → AdamW(weight_decay=1e-4). FID piorou marginalmente (+0.14). O weight decay não beneficiou a UNet neste regime de treino — a regularização implícita do Adam com lr=2e-4 é suficiente para este dataset.

### DDIM v3 — UNet maior (sem melhoria)

Notebook: `ddim_v3.ipynb` | Run: `ddim_v3_largeunet_ep100_seed42` | FID=**65.50**

Única mudança: `hidden_dims=[64,128,256]` → `[64,128,256,256]`. FID piorou significativamente (+20.5). A loss de treino plateou ao mesmo nível que v1 (~0.033), indicando que o bottleneck não é a capacidade do modelo mas o sinal de treino. O bloco extra introduz parâmetros que não convergem adequadamente em 100 epochs para este dataset.

### DDIM v4 — Inference Steps Sweep (melhor resultado)

Notebook: `ddim_v4.ipynb` | Modelo: checkpoint v1 reutilizado (sem retreino)

| Inference steps | FID | Δ vs baseline |
|---|---|---|
| 200 (baseline) | 45.01 | — |
| **250** | **43.18** | **-1.83** ✓ |
| 500 | 49.64 | +4.63 ✗ |

**250 steps é o novo melhor resultado global (FID=43.18).**

O comportamento é esperado: DDIM tem um sweet spot de passos de inferência. Abaixo do óptimo, a ODE é sub-amostrada e a imagem fica ruidosa. Acima, o erro de discretização acumula-se — o modelo foi treinado com 1000 steps de ruído e a sub-sequência de 500 passos diverge numericamente das trajectórias vistas durante o treino.

---

## Resumo de todas as experiências

| Run | FID | KID×10³ | Nota |
|---|---|---|---|
| `dcgan_full_ep100_seed42` | 202.48 | 91.79 | DCGAN baseline — mode collapse |
| `dcgan_v2_ep100_seed42` | — | — | BCE, n_critic=2 — collapse persistiu |
| `dcgan_v3_ep100_seed42` | **102.29** | 84.25 | WGAN-GP — melhor GAN (-50% vs baseline) |
| `ddpm_full_ep100_seed42` | 69.46 | 47.82 | DDPM baseline |
| `ddim_full_ep100_seed42` | 45.01 | 24.96 | DDIM baseline |
| `ddim_full_opt_cosine_ep300` | 57.75 | 32.74 | Cosine schedule + 300 ep — piorou |
| `ddim_full_attn` | 63.68 | 42.82 | Atenção na UNet — piorou |
| `ddim_v2_adamw_ep100_seed42` | 45.15 | 25.05 | AdamW — sem melhoria |
| `ddim_v3_largeunet_ep100_seed42` | 65.50 | 49.30 | UNet maior — piorou |
| `ddim_v4_steps250` | **43.18** | **23.70** | **Melhor resultado global** |
| `ddim_v4_steps500` | 49.64 | 31.70 | 500 steps — piorou |
