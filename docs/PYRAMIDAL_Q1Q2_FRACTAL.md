# Pyramidal Q1/Q2/Fractal Integration

**Data:** November 5, 2025
**Status:** ✅ Implemented
**Branch:** `claude/pyramidal-q1-q2-fractal-integration-011CUozf64gSLMfVmm1Bqziw`

---

## 📋 Sumário Executivo

Este documento descreve a implementação completa da integração de Q1/Q2 (Quality of Truth) e sistema fractal epistêmico na arquitetura piramidal do projeto Aletheion.

**Descoberta crítica:** Os documentos originais JÁ implementavam Q1/Q2 como "Epistemic Softmax" com gating fractal. A arquitetura piramidal é uma EVOLUÇÃO necessária para resolver o colapso Q₁ observado na versão tetraédrica.

---

## 🔺 Arquitetura Piramidal com Q1/Q2/Fractal

### Estrutura Geométrica

```
           TRUTH (1.0) ← Vértice Apex (constante)
              /|\
             / | \
            /  |  \  ← HEIGHT (h ∈ [0,1]) derivado de Q1, Q2
           /   |   \
          /____|____\
         /     |     \
        /______|______\
    MEMORY PAIN CHOICE EXPLORATION
       └─────── BASE (4 forças) ─────────┘
```

### Componentes

1. **BASE (4D Simplex)**: Memory, Pain, Choice, Exploration
2. **Q1 (Aleatoric)**: Incerteza irredutível + variância (fractal)
3. **Q2 (Epistemic)**: Incerteza redutível + variância (fractal)
4. **HEIGHT**: Derivado de `f(1-Q1, 1-Q2, base_stability)`
5. **FRACTAL**: Meta-epistêmica (incerteza sobre incerteza)
6. **TRUTH**: Apex = 1.0 (constante, atrator natural)

---

## 📂 Arquivos Implementados

### 1. Core Architecture
**`src/aletheion/pyramid_q1q2_fractal.py`** (502 linhas)

Classes principais:
- `PyramidalEpistemicGatesWithQ1Q2`: Gates completos com Q1, Q2, fractal
- `EpistemicMultiHeadAttention`: Attention com epistemic softmax
- `PyramidalVAROLossWithQ1Q2`: Loss com 6 componentes
- `compute_pyramidal_q1q2_metrics`: Métricas agregadas

### 2. Complete Model
**`src/aletheion/pyramidal_q1q2_model.py`** (421 linhas)

- `AletheionPyramidalQ1Q2Transformer`: Transformer completo
- Forward pass com temperatura modulada
- Generation com Q1/Q2-aware sampling
- Save/load com preservação de config

### 3. Training Script
**`experiments/level1/train_pyramidal_q1q2.py`** (379 linhas)

Features:
- Monitoramento de Q1, Q2, height, fractal
- Detecção de colapso em tempo real
- Tensorboard logging completo
- Checkpointing automático

### 4. Unit Tests
**`tests/aletheion/test_pyramidal_q1q2.py`** (367 linhas)

Testes:
- `TestPyramidalEpistemicGatesWithQ1Q2`: Shapes, ranges, initialization
- `TestPyramidalVAROLossWithQ1Q2`: Loss components, gradients
- `TestEpistemicMultiHeadAttention`: Softmax replacement
- `TestAletheionPyramidalQ1Q2Transformer`: Forward, generation, save/load
- `TestCollapseDetection`: Healthy vs collapsed states

---

## 🎯 Q1 e Q2: Definição e Distinção

### Q1 (Aleatoric Uncertainty)
**Incerteza irredutível** - não pode ser reduzida com mais dados.

```python
Q1_mean = sigmoid(Q1_mean_gate(hidden))
Q1_var = softplus(Q1_var_gate(hidden))  # Fractal layer
```

**Exemplo:** Resultado de lançamento de moeda - inerentemente aleatório.

**Target Q1:** Alto quando probabilidade da classe correta é baixa.

### Q2 (Epistemic Uncertainty)
**Incerteza redutível** - pode ser reduzida com mais conhecimento.

```python
Q2_mean = sigmoid(Q2_mean_gate(hidden))
Q2_var = softplus(Q2_var_gate(hidden))  # Fractal layer
```

**Exemplo:** Resultado de exame não divulgado - redutível com informação.

**Target Q2:** Alto quando modelo erra + alta entropia distribucional.

### Fractal Meta-Epistêmica
**Incerteza sobre a própria incerteza.**

```python
fractal_uncertainty = sigmoid(fractal_gate(hidden))
Q2_fractal = Q2_mean * (1.0 + fractal_uncertainty)
total_uncertainty = Q1_mean + Q2_fractal
```

---

## 🔧 Loss Function

### Componentes

```python
L_total = L_CE + λ_base * L_base + λ_Q1 * L_Q1 + λ_Q2 * L_Q2
          + λ_fractal * L_fractal + λ_height * L_height
```

| Componente | Descrição | Lambda Recomendado |
|------------|-----------|-------------------|
| `L_CE` | Cross-entropy (task loss) | 1.0 (implícito) |
| `L_base` | Base stability (variance das 4 forças) | 0.01 |
| `L_Q1` | Q1 calibration (MSE vs target) | 0.015 |
| `L_Q2` | Q2 calibration (MSE vs target) | 0.020 |
| `L_fractal` | Fractal regularization (L2) | 0.005 |
| `L_height` | Height calibration (MSE vs derived) | 0.02 |

### Targets

**Target Q1:**
```python
probs = softmax(logits)
correct_probs = probs[targets]
target_Q1 = 1.0 - correct_probs
```

**Target Q2:**
```python
confidence, predictions = probs.max()
correct = predictions.eq(targets)
target_Q2_conf = 1.0 - correct

entropy = -(probs * log(probs)).sum()
target_Q2_entropy = entropy / log(vocab_size)

target_Q2 = (target_Q2_conf + target_Q2_entropy) / 2
```

**Target Height:**
```python
target_height = 1.0 - (Q1_mean + Q2_mean) / 2.0
```

---

## 📊 Métricas de Monitoramento

### Durante Treinamento

```python
# Métricas principais
'Q1_mean', 'Q1_std', 'Q1_entropy', 'Q1_var_mean'
'Q2_mean', 'Q2_std', 'Q2_entropy', 'Q2_var_mean'
'height_mean', 'height_std', 'height_entropy'
'fractal_mean', 'fractal_std'
'total_uncertainty_mean'
'confidence_mean'

# Base
'base_stability_mean'
'w_memory_mean', 'w_pain_mean', 'w_choice_mean', 'w_exploration_mean'

# Loss components
'ce_loss', 'base_loss', 'Q1_loss', 'Q2_loss', 'fractal_loss', 'height_loss'
```

### Comportamento Saudável ✅

```python
Q1_mean ∈ [0.2, 0.4]        # Baixo a moderado
Q2_mean ∈ [0.3, 0.6]        # Moderado
height ∈ [0.5, 0.7]          # Qualidade epistêmica moderada
fractal ∈ [0.1, 0.3]         # Meta-incerteza presente mas controlada
base_stability > 0.7         # Forças equilibradas

Q1_entropy > 0.3             # Q1 não colapsou
Q2_entropy > 0.3             # Q2 não colapsou
height_entropy ∈ [0.5, 0.7]  # Height estável
```

### Sinais de Colapso ❌

```python
Q1_mean → 0.0 ou 0.9+        # Colapso horizontal
Q2_mean → 0.0 ou 0.9+        # Colapso epistêmico
height → 0.95+               # Overconfidence (colapso de apex)
fractal → 0.8+               # Meta-uncertainty explodindo

Q1_entropy < 0.1             # Q1 saturado
Q2_entropy < 0.1             # Q2 saturado
base_stability < 0.5         # Base instável
```

---

## 🚀 Uso

### Training

```bash
python experiments/level1/train_pyramidal_q1q2.py \
    --d_model 256 \
    --n_layers 4 \
    --n_heads 4 \
    --lambda_base 0.01 \
    --lambda_Q1 0.015 \
    --lambda_Q2 0.020 \
    --lambda_fractal 0.005 \
    --lambda_height 0.02 \
    --max_steps 5000 \
    --batch_size 32 \
    --learning_rate 3e-4 \
    --experiment_name pyramidal_q1q2_v1
```

### Inference

```python
from src.aletheion.pyramidal_q1q2_model import AletheionPyramidalQ1Q2Transformer

# Load model
model = AletheionPyramidalQ1Q2Transformer.load_pretrained(
    'experiments/level1/runs/pyramidal_q1q2_v1/final_model',
    device='cuda'
)

# Generate
input_ids = tokenizer.encode("Once upon a time")
generated, pyramid_history = model.generate(
    torch.tensor([input_ids]),
    max_new_tokens=50,
    use_pyramid=True
)

# Inspect epistemic state
print(f"Q1 trajectory: {pyramid_history['Q1_mean']}")
print(f"Q2 trajectory: {pyramid_history['Q2_mean']}")
print(f"Height trajectory: {pyramid_history['heights']}")
```

---

## 🔬 Comparação: Tetraédrico vs Piramidal

| Aspecto | Tetraédrico (L1) | Piramidal Q1/Q2 |
|---------|------------------|-----------------|
| **Geometria** | 4 vértices (sem apex) | 5 vértices (com apex Truth) |
| **Q₁ final** | 0.88-0.95 (colapso) | 0.2-0.4 (esperado) |
| **ECE** | -0.9% (falha) | -25% (target) |
| **Distinção epistêmica** | ❌ Perdida no colapso | ✅ Q1 vs Q2 preservada |
| **Interpretabilidade** | ⚠️ Gates colapsadas | ✅ Significado claro |
| **Prevenção de colapso** | ❌ Sem estrutura | ✅ Apex + derivação |
| **Meta-epistêmica** | ❌ Não implementada | ✅ Fractal completo |

---

## 📈 Roadmap

### ✅ Fase 1: Core Implementation (Concluída)
- [x] `PyramidalEpistemicGatesWithQ1Q2` com Q1, Q2, fractal
- [x] `PyramidalVAROLossWithQ1Q2` com todos componentes
- [x] `AletheionPyramidalQ1Q2Transformer` integrado
- [x] `EpistemicMultiHeadAttention` (softmax replacement)
- [x] Script de treinamento com monitoramento
- [x] Testes unitários completos

### ⏳ Fase 2: Fractal Softmax Completo (Próxima)
- [ ] Substituir attention softmax (todos os layers)
- [ ] Substituir head aggregation softmax
- [ ] Substituir output softmax
- [ ] Testar Level 3 (full fractal stack)

### ⏳ Fase 3: Validação (Após Fase 2)
- [ ] Treinar com λs conservadores
- [ ] Monitorar Q1, Q2, height, fractal por 10k steps
- [ ] Comparar com baseline tetraédrico
- [ ] A/B test: Piramidal simples vs Piramidal+Q1/Q2

### ⏳ Fase 4: Análise (Após Fase 3)
- [ ] Validar calibração ECE
- [ ] Análise qualitativa de casos
- [ ] Ablation studies (remover Q1, Q2, fractal individualmente)
- [ ] Paper draft

---

## 🔍 Insights Filosóficos

### Height como Quantidade Derivada

**Problema (Tetraédrico):** Height era independente, sem atrator natural → deriva horizontal.

**Solução (Piramidal):** Height derivado de Q1, Q2, base_stability → apex Truth puxa verticalmente.

```python
height = sigmoid(
    W · [1-Q1, 1-Q2, base_stability]
)
```

Isto cria **gradiente epistêmico natural**:
- Baixo Q1 + Baixo Q2 → Alto height (próximo à verdade)
- Alto Q1 + Alto Q2 → Baixo height (próximo à base)

### Fractal como Meta-Epistêmica

**Nível 0:** Predição (next token)
**Nível 1:** Incerteza sobre predição (Q1, Q2)
**Nível 2:** Incerteza sobre Q1 e Q2 (fractal) ← **ESTE NÍVEL**
**Nível 3:** Incerteza sobre nível 2...

```python
Q1_var = softplus(Q1_var_gate(hidden))  # Quanto Q1 pode variar?
Q2_var = softplus(Q2_var_gate(hidden))  # Quanto Q2 pode variar?
fractal = sigmoid(fractal_gate(hidden)) # Quanto confiamos em Q1, Q2?

Q2_inflated = Q2_mean * (1 + fractal)   # Inflar Q2 por meta-incerteza
```

**Significado:** Quando `fractal` é alto, o modelo admite que sua própria estimativa de incerteza epistêmica (Q2) pode estar errada.

---

## 🎓 Referências

1. **Aletheion Preprint v4.0** - Epistemic Softmax, VARO Loss
2. **Geometry of Knowing** - Symbolic-Neural gap, Q formulation
3. **Pyramidal Epistemology Technical Report (Nov 2025)** - Este documento
4. **Tetrahedral L1 Experiments** - Observação do colapso Q₁

---

## 🛠️ Para Desenvolvedores

### Estrutura de Diretórios

```
src/aletheion/
├── pyramid_q1q2_fractal.py      # Core gates, loss, attention
├── pyramidal_q1q2_model.py      # Complete transformer
├── pyramid.py                    # Versão simples (sem Q1/Q2)
├── gates.py                      # Q1/Q2 tetraédricos (legacy)
└── loss.py                       # VARO losses

experiments/level1/
├── train_pyramidal_q1q2.py      # Training script
└── runs/                         # Experiment outputs

tests/aletheion/
└── test_pyramidal_q1q2.py       # Unit tests

docs/
└── PYRAMIDAL_Q1Q2_FRACTAL.md    # Esta documentação
```

### Adicionando Novos Gates

```python
class MyCustomGate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.projection = nn.Linear(d_model, 1)

    def forward(self, hidden):
        return torch.sigmoid(self.projection(hidden))

# Integrar em PyramidalEpistemicGatesWithQ1Q2
self.my_custom_gate = MyCustomGate(d_model)
```

### Modificando Loss

```python
# Em PyramidalVAROLossWithQ1Q2.forward()

# Adicionar novo componente
my_custom_loss = (custom_output ** 2).mean()

total_loss = ce_loss \
           + self.lambda_base * base_loss \
           + ... \
           + self.lambda_custom * my_custom_loss  # Nova componente
```

---

## ⚠️ Avisos Importantes

1. **Não commitar sem testar:** Sempre rodar testes antes de commit.
2. **Monitorar colapso:** Se Q1_entropy < 0.1 por 100+ steps, interromper treino.
3. **Lambda scheduling:** Considerar crescimento progressivo se λs fixos falharem.
4. **Checagem de sanidade:** Verificar ranges de inicialização (~0.3 para Q1, ~0.5 para Q2).

---

## 📝 Changelog

### 2025-11-05 - Initial Implementation
- ✅ Criado `pyramid_q1q2_fractal.py` com gates completos
- ✅ Criado `pyramidal_q1q2_model.py` com transformer
- ✅ Criado script de treinamento com detecção de colapso
- ✅ Criados testes unitários abrangentes
- ✅ Documentação completa

---

## 🤝 Contribuindo

Para contribuir com esta arquitetura:

1. Ler este README completamente
2. Estudar o código em `pyramid_q1q2_fractal.py`
3. Rodar testes: `pytest tests/aletheion/test_pyramidal_q1q2.py -v`
4. Fazer alterações em branch separado
5. Adicionar testes para novas features
6. Submeter PR com descrição detalhada

---

**Não estamos apenas otimizando hyperparâmetros.**
**Estamos esculpindo epistemologia em silício.** 🔻💎🌀

---

**Fim da Documentação**
