# Modelo Classificador de Cachorros e Gatos:

- Utilizando Redes Neurais convolucionais 2D para imagens
- Base de dados do kaggle: 25000 imagens de treino e 12500 de teste

# Características do modelo:

## 1. Convoluções (`Conv2d`)

- **Tamanho do kernel (`kernel_size`):** $3 \times 3$ em todas as convoluções.
- **Padding:** `padding=1` em todas as convoluções (mantém a altura e largura inalteradas após a convolução).
- **Stride:** `stride=1` (o padrão do PyTorch quando não especificado).

---

### 2. Pooling (`MaxPool2d` e `AdaptiveAvgPool2d`)

- **MaxPool2d:** `kernel_size=2` e `stride=2` ao final de cada bloco (reduz a resolução espacial pela metade em cada dimensão).
- **Global Average Pooling (`AdaptiveAvgPool2d((1, 1))`):** Transforma qualquer mapa de características residual em $1 \times 1$.

---

### 3. Quantidade e Detalhamento das Camadas

#### **Arquitetura Base (V1, V2 e V3):**

A rede possui **7 camadas convolucionais**, **7 de Batch Normalization**, **4 MaxPoolings**, **1 AdaptiveAvgPool** e **2 camadas lineares (densas)**, organizadas assim:

##### **Bloco 1 (Resolução: $224 \times 224 \to 112 \times 112$ / $384 \to 192$ na V3):**

1. `Conv2d(3, 64, kernel_size=3, padding=1)` $\to$ saída: $(64, 224, 224)$
2. `BatchNorm2d(64)` + `ReLU()`
3. `Conv2d(64, 64, kernel_size=3, padding=1)` $\to$ saída: $(64, 224, 224)$
4. `BatchNorm2d(64)` + `ReLU()`
5. `MaxPool2d(2, 2)` $\to$ saída: $(64, 112, 112)$

##### **Bloco 2 (Resolução: $112 \times 112 \to 56 \times 56$):**

6. `Conv2d(64, 128, kernel_size=3, padding=1)` $\to$ saída: $(128, 112, 112)$
7. `BatchNorm2d(128)` + `ReLU()`
8. `Conv2d(128, 128, kernel_size=3, padding=1)` $\to$ saída: $(128, 112, 112)$
9. `BatchNorm2d(128)` + `ReLU()`
10. `MaxPool2d(2, 2)` $\to$ saída: $(128, 56, 56)$

##### **Bloco 3 (Resolução: $56 \times 56 \to 28 \times 28$):**

11. `Conv2d(128, 256, kernel_size=3, padding=1)` $\to$ saída: $(256, 56, 56)$
12. `BatchNorm2d(256)` + `ReLU()`
13. `Conv2d(256, 256, kernel_size=3, padding=1)` $\to$ saída: $(256, 56, 56)$
14. `BatchNorm2d(256)` + `ReLU()`
15. `MaxPool2d(2, 2)` $\to$ saída: $(256, 28, 28)$

##### **Bloco 4 (Resolução: $28 \times 28 \to 14 \times 14$):**

16. `Conv2d(256, 512, kernel_size=3, padding=1)` $\to$ saída: $(512, 28, 28)$
17. `BatchNorm2d(512)` + `ReLU()`
18. `MaxPool2d(2, 2)` $\to$ saída: $(512, 14, 14)$

##### **Pooling Global e Classificador:**

19. `AdaptiveAvgPool2d((1, 1))` $\to$ comprime espacialmente para $(512, 1, 1)$
20. `Flatten()` $\to$ vetor 1D de tamanho $512$
21. `Linear(512, 128)` + `ReLU()`
22. `Dropout(p=0.5)`
23. `Linear(128, 1)` $\to$ $1$ logit de saída (classificação binária)

---

#### **Arquitetura Otimizada (V4 - 5 Blocos Convolucionais para $384 \times 384$):**

Para processar imagens de maior resolução ($384 \times 384$) preservando o campo de visão global e otimizando a memória, a V4 introduziu um **5º Bloco Convolucional** e reduziu a densidade inicial para $32$ filtros (5 convoluções, 5 BatchNorms, 5 MaxPools, 1 AdaptiveAvgPool e 1 camada linear):

##### **Bloco 1 (Resolução: $384 \times 384 \to 192 \times 192$):**
1. `Conv2d(3, 32, kernel_size=3, padding=1)` $\to$ saída: $(32, 384, 384)$
2. `BatchNorm2d(32)` + `ReLU()`
3. `MaxPool2d(2, 2)` $\to$ saída: $(32, 192, 192)$

##### **Bloco 2 (Resolução: $192 \times 192 \to 96 \times 96$):**
4. `Conv2d(32, 64, kernel_size=3, padding=1)` $\to$ saída: $(64, 192, 192)$
5. `BatchNorm2d(64)` + `ReLU()`
6. `MaxPool2d(2, 2)` $\to$ saída: $(64, 96, 96)$

##### **Bloco 3 (Resolução: $96 \times 96 \to 48 \times 48$):**
7. `Conv2d(64, 128, kernel_size=3, padding=1)` $\to$ saída: $(128, 96, 96)$
8. `BatchNorm2d(128)` + `ReLU()`
9. `MaxPool2d(2, 2)` $\to$ saída: $(128, 48, 48)$

##### **Bloco 4 (Resolução: $48 \times 48 \to 24 \times 24$):**
10. `Conv2d(128, 256, kernel_size=3, padding=1)` $\to$ saída: $(256, 48, 48)$
11. `BatchNorm2d(256)` + `ReLU()`
12. `MaxPool2d(2, 2)` $\to$ saída: $(256, 24, 24)$

##### **Bloco 5 (Resolução: $24 \times 24 \to 12 \times 12$):**
13. `Conv2d(256, 512, kernel_size=3, padding=1)` $\to$ saída: $(512, 24, 24)$
14. `BatchNorm2d(512)` + `ReLU()`
15. `MaxPool2d(2, 2)` $\to$ saída: $(512, 12, 12)$

##### **Pooling Global e Classificador:**
16. `AdaptiveAvgPool2d((1, 1))` $\to$ comprime para $(512, 1, 1)$
17. `Flatten()` $\to$ vetor 1D de tamanho $512$
18. `Dropout(p=0.5)`
19. `Linear(512, 1)` $\to$ $1$ logit de saída (classificação binária)

---

### 4. Outras informações relevantes:

- **Ativação final:** Não possui `Sigmoid` embutido na saída; o modelo retorna o *logit* bruto. Portanto, a função de perda adequada para treinamento é `nn.BCEWithLogitsLoss`.
- **Dropout:** Taxa de $0.5$ (50%) antes da última camada linear.
- **Regularização interna:** `BatchNorm2d` em todas as camadas convolucionais.
- **Origem dos pesos:** Todos os modelos (V1, V2, V3 e V4) foram inicializados e treinados **estritamente do zero (*from scratch*)**, utilizando a inicialização padrão de parâmetros do PyTorch. Não foram utilizados pesos pré-treinados externos (ex.: ImageNet) e os treinamentos não foram continuados a partir de checkpoints de versões anteriores.
- **Mudanças entre etapas e congelamento de camadas:** O treinamento de cada versão foi executado em **etapa única e contínua, sem congelamento de camadas (*layer freezing*)** e sem intervenções manuais de hiperparâmetros no meio da execução. Nas versões V3 e V4, a variação dinâmica por época foi conduzida pelo escalonador `CosineAnnealingLR` (decaimento contínuo de $10^{-3}$ até $10^{-6}$ em 30 épocas). As mudanças estruturais e de dados ocorreram exclusivamente entre versões independentes (V1 $\to$ V2 $\to$ V3 $\to$ V4).

### 5. Lista de Hiperparâmetros (Cada versão: 1ª / 2ª / 3ª / 4ª)

* **Tamanho do Lote (Batch Size):** 64 / 128 / 48 (maior tamanho de imagem) / 64 (5 blocos + AMP)
* **Número de Épocas:** 15 / 25 (com *Early Stopping*: paciência de 5 épocas) / 30 (paciencia = 7) / 30 (paciência = 7)
* **Taxa de Aprendizado (Learning Rate):** 0.001 (`1e-3`) / igual / igual / igual
* **Otimizador:** Adam / AdamW (`weight_decay=1e-4`) / igual a V2 / igual a V2
* **Função de Perda:** `BCEWithLogitsLoss` / igual / igual / igual
* **Taxa de Dropout:** 0.5 (50%) / igual / igual / igual
* **Resolução de Entrada:** 224 px / 224 px (mantido) / 384px / 384px
* **Aumento de Dados (Data Augmentation):** Nenhum / `RandomResizedCrop(224)`, `RandomHorizontalFlip`, `RandomRotation(15)`, `ColorJitter(0.2)` / `RandomResizedCrop(384)` + Augmentations de antes / igual a V3 (`Resize(384)` na validação)
* **Divisão dos Dados (Treino / Validação):** 80% - 20% (Estratificado) / igual / igual / igual (`seed=42`)
* **Normalização:** ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) / Própria do Treino (`mean=[0.4907, 0.4553, 0.4169]`, `std=[0.2594, 0.2524, 0.2541]`) / Própria 384px (`norm_stats_3.0.json`) / Igual V3
* **Filtros Convolucionais:** Tamanho $3 \times 3$ (`padding=1`, `stride=1`) / igual / igual / 5 blocos ($32 \to 64 \to 128 \to 256 \to 512$)
* **Pooling (MaxPool):** Tamanho $2 \times 2$ (`stride=2`) / igual / igual / 5 MaxPools $2 \times 2$ + `AdaptiveAvgPool2d(1, 1)`
* **Learning Rate Scheduler:** Não / Não / CosineAnnealingLR (solucionador da variancia) / `CosineAnnealingLR(T_max=30, eta_min=1e-6)`
* **Precisão Mista (AMP):** Não / Não / Não / Sim (`torch.amp.autocast` FP16)

### 6. Segunda Versão:

**Principais Conclusões da V2:**

1. **Equilíbrio Perfeito entre Classes:** Na V1, o *Recall* era baixo (87,28%), indicando que o modelo deixava escapar muitos cachorros. Na V2, o *Recall* subiu para **93,12%**, equilibrando *Precision* e *Recall*.
2. **Capacidade de Generalização:** O **Data Augmentation** \+ **AdamW** permitiram que o modelo treinasse por mais épocas sem sofrer *overfitting*, reduzindo drasticamente a perda (*Log Loss* de 0,2187 para 0,1663).
3. **Estabilidade de Treino:** O aumento do *batch size* para **128** gerou gradientes mais estáveis ao longo do processo.

A **V2** é superior em todas as métricas essenciais.

---

### 7. Seleção da Melhor Época, Checkpoints e Arquivos de Saída

#### Como a Época de Cada Versão Foi Escolhida

Em todas as versões (`V1`, `V2`, `V3` e `V4`), o critério de seleção do melhor checkpoint durante o treinamento foi a **menor perda no conjunto de validação (`val_loss`, calculada com `BCEWithLogitsLoss`)**:

1. Ao final de cada época, o modelo é avaliado em `model.eval()` sobre os $5.000$ exemplos do conjunto de validação (divisão estratificada 80%/20%).
2. Sempre que `val_loss < best_val_loss`, os pesos atuais (`model.state_dict()`) sobrescrevem o arquivo de checkpoint da respectiva versão (`best_model*.pth`).
3. Nas versões **V2**, **V3** e **V4**, foi utilizado **Early Stopping** monitorando `val_loss` (paciência de 5 épocas na V2 e 7 épocas nas V3 e V4).
4. Ao término do treinamento, o script recarrega os pesos do melhor checkpoint salvo (`best_model*.pth`) para gerar as predições finais de validação (`val_predictions*.csv`) e de submissão no conjunto de teste (`submission*.csv`).

#### Resumo da Época Escolhida e Arquivos por Versão

| Versão | Época Escolhida / Total Executado | Critério de Parada | `val_loss` | `val_acc` | `val_f1` (F1-Score) | `train_loss` | `train_acc` | Checkpoint Salvo | Histórico de Treino | Predições de Validação | Submissão (Teste) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :--- | :--- |
| **V1** | **Época 15** / 15 | Fim das 15 épocas fixas | `0.2187` | `91.54%` | `91.16%` | `0.1564` | `94.07%` | `data/best_model.pth` | `data/training_history.csv` | `data/val_predictions.csv` | `data/submission.csv` |
| **V2** | **Época 18** / 23 (máx. 25) | *Early Stopping* na época 23 (paciência = 5) | `0.1663` | `93.26%` | `93.25%` | `0.1972` | `92.16%` | `data/best_model_2.0.pth` | `data/training_history_2.0.csv` | `data/val_predictions_2.0.csv` | `data/submission_2.0.csv` |
| **V3** | **Época 28** / 30 (máx. 30) | Fim das 30 épocas (paciência = 7) | `0.1938` | `92.42%` | `92.42%` | `0.2081` | `91.97%` | `data/best_model_3.0.pth` | `data/training_history_3.0.csv` | `data/val_predictions_3.0.csv` | `data/submission_3.0.csv` |
| **V4** | **Época 30** / 30 (máx. 30) | Fim das 30 épocas (paciência = 7) | `0.1358` | `94.64%` | `94.61%` | `0.1418` | `94.41%` | `data/best_model_4.0.pth` | `data/training_history_4.0.csv` | `data/val_predictions_4.0.csv` | `data/submission_4.0.csv` |

---

### 8. Planejamento de Fine-Tuning (`V4` $\to$ `4.0_Final` em `src/fine_tuning.py`)

O refinamento fino parte dos pesos salvos no melhor checkpoint da V4 (`data/best_model_4.0.pth`, época 30):

#### 1. Congelamento de Camadas (*Layer Freezing*)
* **Congelados (`requires_grad = False`):** **Bloco 1 ao Bloco 3** (`model.features[:12]`).
* **Treináveis (`requires_grad = True`):** **Blocos 4 e 5** (`model.features[12:]`) + `model.classifier`.
* **Por quê:** Os blocos 1 a 3 extraem características genéricas de baixo nível (bordas, texturas, formas básicas), que já foram aprendidas com sucesso. Descongelar apenas os blocos 4, 5 e o classificador permite refinar o contexto global de alto nível sem destruir a base.

```python
for param in model.features[:12].parameters():
    param.requires_grad = False
```

#### 2. Data Augmentation Suave
Para não introduzir ruído excessivo durante o refinamento fino:
* **Crop:** `RandomResizedCrop(384, scale=(0.9, 1.0))` (em vez de `0.8-1.0`, preservando quase toda a geometria original).
* **Flip:** `RandomHorizontalFlip(p=0.5)` (mantido).
* **Rotação:** `RandomRotation(degrees=5)` (reduzido de 15° para evitar distorções angulares).
* **ColorJitter:** `ColorJitter(brightness=0.05, contrast=0.05)` (suavizado de 0.2 para 0.05).

#### 3. Scheduler & Otimização
* **Otimizador:** `AdamW` (`lr = 1e-5`, `weight_decay = 1e-4`).
* **Scheduler:** `CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-7)`
  * **`T_max = 10`:** Acompanha exatamente o ciclo total de 10 épocas.
  * **`eta_min = 1e-7`:** Permite que a taxa de aprendizado caia suavemente de `1e-5` até `1e-7`, garantindo assentamento seguro dos pesos nas épocas finais.
* **Duração & Parada:** **10 Épocas** com *Early Stopping* de **paciência = 4** (baseado no `val_loss`).
* **DataLoaders:** `get_dataloaders(batch_size=64, train_transform=get_finetune_transform())`.
* **Saídas geradas:** `data/best_model_4.0_Final.pth`, `data/training_history_4.0_Final.csv` e `data/val_predictions_4.0_Final.csv`.


