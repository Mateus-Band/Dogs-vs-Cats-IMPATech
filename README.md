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

A rede possui **7 camadas convolucionais**, **7 de Batch Normalization**, **4 MaxPoolings**, **1 AdaptiveAvgPool** e **2 camadas lineares (densas)**, organizadas assim:

#### **Bloco 1 (Resolução: $224 \times 224 \to 112 \times 112$):**

1. `Conv2d(3, 64, kernel_size=3, padding=1)` $\to$ saída: $(64, 224, 224)$
2. `BatchNorm2d(64)` + `ReLU()`
3. `Conv2d(64, 64, kernel_size=3, padding=1)` $\to$ saída: $(64, 224, 224)$
4. `BatchNorm2d(64)` + `ReLU()`
5. `MaxPool2d(2, 2)` $\to$ saída: $(64, 112, 112)$

#### **Bloco 2 (Resolução: $112 \times 112 \to 56 \times 56$):**

6. `Conv2d(64, 128, kernel_size=3, padding=1)` $\to$ saída: $(128, 112, 112)$
7. `BatchNorm2d(128)` + `ReLU()`
8. `Conv2d(128, 128, kernel_size=3, padding=1)` $\to$ saída: $(128, 112, 112)$
9. `BatchNorm2d(128)` + `ReLU()`
10. `MaxPool2d(2, 2)` $\to$ saída: $(128, 56, 56)$

#### **Bloco 3 (Resolução: $56 \times 56 \to 28 \times 28$):**

11. `Conv2d(128, 256, kernel_size=3, padding=1)` $\to$ saída: $(256, 56, 56)$
12. `BatchNorm2d(256)` + `ReLU()`
13. `Conv2d(256, 256, kernel_size=3, padding=1)` $\to$ saída: $(256, 56, 56)$
14. `BatchNorm2d(256)` + `ReLU()`
15. `MaxPool2d(2, 2)` $\to$ saída: $(256, 28, 28)$

#### **Bloco 4 (Resolução: $28 \times 28 \to 14 \times 14$):**

16. `Conv2d(256, 512, kernel_size=3, padding=1)` $\to$ saída: $(512, 28, 28)$
17. `BatchNorm2d(512)` + `ReLU()`
18. `MaxPool2d(2, 2)` $\to$ saída: $(512, 14, 14)$

#### **Pooling Global e Classificador:**

19. `AdaptiveAvgPool2d((1, 1))` $\to$ comprime espacialmente para $(512, 1, 1)$
20. `Flatten()` $\to$ vetor 1D de tamanho $512$
21. `Linear(512, 128)` + `ReLU()`
22. `Dropout(p=0.5)`
23. `Linear(128, 1)` $\to$ $1$ logit de saída (classificação binária)

---

### 4. Outras informações relevantes:

- **Ativação final:** Não possui `Sigmoid` embutido na saída; o modelo retorna o *logit* bruto. Portanto, a função de perda adequada para treinamento é `nn.BCEWithLogitsLoss`.
- **Dropout:** Taxa de $0.5$ (50%) antes da última camada linear.
- **Regularização interna:** `BatchNorm2d` em todas as camadas convolucionais.

### 5. Lista de Hiperparâmetros (Cada versão: 1ª / 2ª / ...)

* **Tamanho do Lote (Batch Size):** 64 / 128 / 64
* **Número de Épocas:** 15 / 25 (com *Early Stopping*: paciência de 5 épocas) / 30 (paciencia = 7)
* **Taxa de Aprendizado (Learning Rate):** 0.001 (`1e-3`) / igual / igual
* **Otimizador:** Adam / AdamW (`weight_decay=1e-4`) / igual a V2
* **Função de Perda:** `BCEWithLogitsLoss` / igual / igual
* **Taxa de Dropout:** 0.5 (50%) / igual / igual
* **Resolução de Entrada:** 224 px / 224 px (mantido) / 384px
* **Aumento de Dados (Data Augmentation):** Nenhum / `RandomResizedCrop(224)`, `RandomHorizontalFlip`, `RandomRotation(15)`, `ColorJitter(0.2)`/ `RandomResizedCrop(384)`+ Augmentations de antes.
* **Divisão dos Dados (Treino / Validação):** 80% - 20% (Estratificado) / igual /
* **Normalização:** ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) / Própria do Treino (`mean=[0.4907, 0.4553, 0.4169]`, `std=[0.2594, 0.2524, 0.2541]`) / Igual V2
* **Filtros Convolucionais:** Tamanho $3 \times 3$ (`padding=1`, `stride=1`) / igual
* **Pooling (MaxPool):** Tamanho $2 \times 2$ (`stride=2`) / igual
* **Learning Rate Scheduler:** Não / Não / CosineAnnealingLR (solucionador da variancia)

### 6. Segunda Versão:

**Principais Conclusões da V2:**

1. **Equilíbrio Perfeito entre Classes:** Na V1, o *Recall* era baixo (87,28%), indicando que o modelo deixava escapar muitos cachorros. Na V2, o *Recall* subiu para **93,12%**, equilibrando *Precision* e *Recall*.
2. **Capacidade de Generalização:** O **Data Augmentation** \+ **AdamW** permitiram que o modelo treinasse por mais épocas sem sofrer *overfitting*, reduzindo drasticamente a perda (*Log Loss* de 0,2187 para 0,1663).
3. **Estabilidade de Treino:** O aumento do *batch size* para **128** gerou gradientes mais estáveis ao longo do processo.

A **V2** é superior em todas as métricas essenciais.
