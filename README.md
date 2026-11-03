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
