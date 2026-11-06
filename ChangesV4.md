# Mudanças da Versão 4 (V4)

## 1. Mudanças na Estrutura da Rede (`src/model.py`)

Para processar imagens em $384 \times 384$ sem perder o campo de visão (contexto global), foi adicionado o 5º Bloco Convolucional:

* **Bloco 1:** `Conv2d(3 -> 32)` + `BatchNorm2d` + `ReLU` + `MaxPool2d(2x2)` $\rightarrow$ Mapa: $192 \times 192$
* **Bloco 2:** `Conv2d(32 -> 64)` + `BatchNorm2d` + `ReLU` + `MaxPool2d(2x2)` $\rightarrow$ Mapa: $96 \times 96$
* **Bloco 3:** `Conv2d(64 -> 128)` + `BatchNorm2d` + `ReLU` + `MaxPool2d(2x2)` $\rightarrow$ Mapa: $48 \times 48$
* **Bloco 4:** `Conv2d(128 -> 256)` + `BatchNorm2d` + `ReLU` + `MaxPool2d(2x2)` $\rightarrow$ Mapa: $24 \times 24$
* **Bloco 5 (NOVO):** `Conv2d(256 -> 512)` + `BatchNorm2d` + `ReLU` + `MaxPool2d(2x2)` $\rightarrow$ Mapa: $12 \times 12$
* **Pooling Global:** `AdaptiveAvgPool2d((1, 1))` $\rightarrow$ Reduz o mapa para vetor de 512 canais.
* **Classificador (Linear):** Entrada ajustada para 512 $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `Linear(512, 1)`.

---

## 2. Mudanças nos Hiperparâmetros (`src/dataset.py` e `src/train.py`)

* **Resolução de Entrada:** $384 \times 384$
  * `RandomResizedCrop(384, scale=(0.8, 1.0))` no treino e `Resize((384, 384))` na validação.
* **Tamanho do Lote (Batch Size):** `64`
  * Equilíbrio para a VRAM da GPU com imagens de 384px e 512 canais.
* **Número de Épocas:** `30` (com *Early Stopping* de paciência = `7`).
* **Otimizador:** `AdamW(lr=0.001, weight_decay=1e-4)`.
* **LR Scheduler:** `CosineAnnealingLR`
  * `scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)`
  * Chamado 1x por época no final de cada loop de treinamento (`scheduler.step()`).
* **Data Augmentation:** Mantido (`RandomResizedCrop(384, scale=(0.8, 1.0))` + `RandomHorizontalFlip()` + `RandomRotation(15)` + `ColorJitter(brightness=0.2, contrast=0.2)`).
* **Normalização:** Própria do dataset de treino (`data/norm_stats_3.0.json`).

---

## 3. Garantias Adicionais de Treinamento e Execução

### 3.1. Uso de Precisão Mista (`torch.amp`) — Aceleração Grátis
* **Configuração:** Ativado `torch.amp.autocast` e `torch.amp.GradScaler` durante o treino em GPU (`cuda`).
* **Por quê:** Como imagens de $384 \times 384$ exigem mais processamento, a precisão mista (FP16) reduz significativamente o tempo por época na RTX A4500 e economiza VRAM, mantendo a mesma precisão final.

### 3.2. Transformações Distintas de Treino e Validação
* **Configuração:**
  * **Treino:** `RandomResizedCrop(384, scale=(0.8, 1.0))` + Data Augmentations + `Normalize`.
  * **Validação:** Apenas `Resize((384, 384))` estático (sem giros ou cortes aleatórios) + `Normalize`.
* **Por quê:** A validação precisa medir a capacidade da rede em imagens limpas e consistentes em todas as épocas.

### 3.3. Otimização do `DataLoader` (`num_workers` e `pin_memory`)
* **Configuração:** Definido `num_workers=4` e `pin_memory=True` no `DataLoader` (com transferência `non_blocking=True`).
* **Por quê:** Imagens de 384px exigem mais da CPU para descompactar e aplicar transformações. Evita que a GPU fique ociosa esperando pela CPU (gargalo de I/O).

### 3.4. Fixar a Semente Aleatória (*Random Seed*)
* **Configuração:** `set_seed(42)` (`random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`) no início do `train.py`.
* **Por quê:** Garante que qualquer diferença na acurácia venha realmente das melhorias de arquitetura/hiperparâmetros, e não da sorte no embaralhamento dos lotes.

### 3.5. Salvamento Baseado em `Val Loss`
* **Configuração:** O arquivo `best_model_4.0.pth` continua sendo salvo exclusivamente no momento da **menor `val_loss`** (e não necessariamente da última época).

