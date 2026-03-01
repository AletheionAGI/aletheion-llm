# Aletheion-LLM - Avaliacao Financeira (Valuation)

**Data:** 2026-02-28
**Versao:** 1.0
**Metodologia:** Auditoria automatizada de codebase + analise de mercado

---

## 1. Executive Summary

O aletheion-llm e uma implementacao em PyTorch de "epistemic softmax" para LLMs
com calibracao de incerteza, acompanhada de um paper academico e infraestrutura
de treinamento/avaliacao.

| Cenario | Faixa de Valor (USD) |
|---------|---------------------|
| **Custo de Reposicao** | US$ 25K - US$ 65K |
| **IP (Propriedade Intelectual)** | US$ 15K - US$ 50K |
| **Valor Estrategico (como componente)** | US$ 30K - US$ 80K |
| **Valor Justo Estimado** | **US$ 25K - US$ 60K** |

---

## 2. Inventario Tecnico

### 2.1 Codigo Fonte

| Componente | Arquivos | Linhas | Observacao |
|------------|----------|--------|-----------|
| src/ (core Python) | 13 | 3,636 | Implementacao funcional, zero stubs |
| tests/ | 7 | 1,207 | Testes reais com assertions |
| experiments/ | 10 | 7,911 | Scripts de treinamento funcionais |
| examples/ | 6 | 1,037 | Scripts de uso |
| scripts/ (shell) | 9 | 667 | Automacao |
| data/ | 3 | ~350 | Dataset utilities |
| config/ (YAML) | 4 | 253 | Configs de modelo |
| hf_space/ | ~20 | ~5,700 | Demo Gradio (duplica src/) |
| **Total Python** | **52** | **17,411** | |

### 2.2 Documentacao e Paper

| Componente | Arquivos | Linhas |
|------------|----------|--------|
| Paper LaTeX | ~20 | 4,738 |
| Docs Markdown | 16 | 10,698 |
| Root Markdown (README, CHANGELOG, etc.) | 12 | ~4,000 |
| **Total documentacao** | **~48** | **~19,436** |

### 2.3 Outputs de Treinamento

| Artefato | Tamanho | Conteudo |
|----------|---------|----------|
| outputs/baseline/ | ~2.7 MB | history.json (treinamento real) |
| outputs/pyramidal/ | ~20.5 MB | history.json (treinamento real) |
| outputs/comparison/ | ~100 KB | results.json + plots |

### 2.4 Metricas Estruturais

| Metrica | Valor |
|---------|-------|
| Commits totais | 271 |
| Periodo de desenvolvimento | 10 dias (04-13/nov/2025) |
| Commits por autor humano | 168 (62%) |
| Commits por AI (Claude) | 101 (37%) |
| Outros | 2 (1%) |
| Ratio documentacao:codigo | **5.3:1** |
| Linhas genuinamente novas (epistemic logic) | **~900** |
| Linhas standard boilerplate (transformer/training) | ~2,700 |
| Modelo pre-treinado distribuido | Nao |
| Dependencias externas | 13 (PyTorch, HF Transformers, etc.) |

---

## 3. Analise Tecnica

### 3.1 O que e Real e Funcional

1. **Implementacao PyTorch completa** (3,636 LOC em src/) - zero stubs, zero placeholders
2. **Testes com assertions reais** (1,207 LOC, ~30 testes)
3. **Scripts de treinamento funcionais** (7,911 LOC em experiments/)
4. **Historicos de treinamento genuinos** (23+ MB de dados)
5. **Paper academico completo** (4,738 linhas LaTeX, ~200 refs)
6. **Estrutura profissional** (CI/CD, pre-commit, packaging, type hints)

### 3.2 Componentes Tecnicos (src/)

| Arquivo | Linhas | Funcao |
|---------|--------|--------|
| pyramid_q1q2_fractal.py | 661 | Gates epistemicos com Q1/Q2 + fractal variance |
| loss.py | 522 | VaroLoss + PyramidalVAROLoss + metricas calibracao |
| pyramidal_q1q2_model.py | 451 | Transformer completo com Q1/Q2/Fractal |
| pyramidal_model.py | 413 | Transformer com piramide epistemica |
| model.py (aletheion) | 407 | AletheionTransformer Level 1 |
| pyramid.py | 319 | PyramidalEpistemicGates (4 forcas + altura) |
| gates.py | 282 | Q1 gate, Q2 gate, epistemic_softmax |
| model.py (base) | 231 | BaselineTransformer (GPT-2 style) |
| attention.py | 165 | MultiHeadAttention + CausalSelfAttention |
| utils.py | 90 | LR schedulers, seed, device |
| tokenizer.py | 17 | Wrapper GPT2TokenizerFast |

### 3.3 Avaliacao da Novidade Tecnica

#### O que a "epistemic softmax" realmente faz:

```
p_gated = confidence * softmax(logits / temperature) + (1 - confidence) * uniform
```

Onde:
- **Q1 gate** = `Linear(d_model, 1) + Sigmoid` (um unico layer linear)
- **Q2 gate** = Multi-head attention pooling + linear + sigmoid
- **confidence** = Q1 * Q2 (escalar [0,1])
- **Piramide** = `Linear(d_model, 4) + Softmax` (4 "forcas" nomeadas)
- **Fractal** = `Softplus` sobre variancias (Q1_var, Q2_var)

#### Avaliacao de originalidade:

| Claim | Realidade | Originalidade |
|-------|-----------|---------------|
| "Epistemic softmax" | Mixture de softmax + distribuicao uniforme gated por confianca aprendida | **Baixa-Media** - variante de label smoothing / confidence calibration |
| "Piramide epistemica" | Softmax 4D + escalar altura derivado de estatisticas do hidden state | **Baixa** - naming convention sobre operacoes lineares padrao |
| "Fractal" | Softplus sobre variancias | **Baixa** - nao implementa estrutura fractal real |
| "VARO loss" | Cross-entropy + MSE entre incerteza predita e alvo | **Media** - combinacao util, embora nao inedita |
| Q1 + Q2 decomposicao | Separacao aleatoric/epistemic via gates aprendidos | **Media** - decomposicao conhecida na literatura, implementacao especifica e original |

### 3.4 Resultados Experimentais

**Dados reais de `outputs/comparison/results.json`:**

| Metrica | Baseline | Pyramidal | Melhor? |
|---------|----------|-----------|---------|
| ECE (Expected Calibration Error) | **0.009** | 0.015 | Baseline |
| Perplexidade | Similar | Similar | Empate |
| p-value (ECE) | - | 0.935 | Nao significativo |

**Discrepancia critica:** O paper afirma melhoria de 89% no ECE (0.104 -> 0.011),
mas os dados no repositorio mostram que o modelo piramidal performou **pior** que
o baseline (ECE 0.015 vs 0.009, p=0.935 nao significativo).

### 3.5 Issues Identificados

| # | Issue | Severidade |
|---|-------|-----------|
| 1 | Paper claims nao correspondem aos dados experimentais | **Alta** |
| 2 | "Pyramidal/Fractal" sao nomes marketing sobre operacoes simples | Media |
| 3 | 37% dos commits gerados por AI (Claude) | Media |
| 4 | Headers "PROPRIETARY" em arquivos distribuidos sob AGPL | Media |
| 5 | Nenhum modelo pre-treinado distribuido | Media |
| 6 | GIT_HISTORY_EVIDENCE.txt referencia repo diferente (AletheionGuard) | Media |
| 7 | hf_space/ duplica ~3,500 linhas de src/ | Baixa |
| 8 | Ratio documentacao:codigo de 5.3:1 (inflacionado) | Baixa |
| 9 | HF Space demo usa pesos aleatorios (nao treinados) | Media |

---

## 4. Licenciamento

| Licenca | Escopo |
|---------|--------|
| AGPL-3.0-or-later | Open source (uso geral) |
| Commercial License | Uso comercial (California law) |

**Contradicao:** Arquivos em `src/aletheion/` marcados "PROPRIETARY AND CONFIDENTIAL -
LEVEL 3 PROPRIETARY ARCHITECTURE" estao distribuidos publicamente sob AGPL-3.0.
Isso cria ambiguidade juridica que necessita resolucao.

---

## 5. Metodologia de Avaliacao

### 5.1 Custo de Reposicao

| Fator | Valor |
|-------|-------|
| LOC produtivo novel (~900 linhas epistemic) | ~12-18 dias-engenheiro |
| LOC boilerplate (~2,700 linhas transformer) | ~20-35 dias-engenheiro |
| Testes + experiments (~9,100 linhas) | ~30-50 dias-engenheiro |
| Paper LaTeX (~4,700 linhas) | ~15-25 dias-pesquisador |
| Docs + config + scripts | ~10-15 dias |
| **Total dias** | **87-143 dias** |
| Custo diario (ML engineer senior, USD) | $400 - $800 |
| **Custo de reposicao** | **US$ 35K - US$ 114K** |

**Desconto (resultados nao validados, -40%):**

| Cenario | Total |
|---------|-------|
| Conservador | **US$ 21K** |
| Moderado | **US$ 45K** |
| Otimista | **US$ 68K** |

### 5.2 Valor de IP

| Ativo | Valor |
|-------|-------|
| Epistemic softmax (Q1/Q2 gating, ~900 LOC) | US$ 5K - 15K |
| VARO loss function | US$ 2K - 5K |
| Paper academico (4,738 linhas LaTeX) | US$ 5K - 20K |
| Dados de treinamento/outputs (23MB) | US$ 1K - 3K |
| Infraestrutura de benchmark (experiments/) | US$ 2K - 7K |
| **Total IP** | **US$ 15K - US$ 50K** |

**Nota:** O valor de IP e limitado por:
- Resultados experimentais nao comprovam as claims do paper
- A tecnica core (confidence-gated softmax) tem prior art na literatura
- Nao ha patente registrada
- Codigo distribuido publicamente sob AGPL (qualquer um pode usar)

### 5.3 Potencial de Receita

| Canal | Projecao Anual |
|-------|---------------|
| Licenca comercial (se resultados validados) | US$ 5K - 20K/cliente |
| Consultoria em uncertainty calibration | US$ 10K - 30K |
| Integracao como modulo em plataformas maiores | US$ 15K - 50K |

**Projecao e altamente especulativa** sem validacao experimental positiva.

### 5.4 Comparaveis

| Projeto Open Source Similar | Status | Referencia |
|----------------------------|--------|-----------|
| Laplace Redux (uncertainty in NNs) | Academico, gratuito | ~5K stars GitHub |
| Uncertainty Baselines (Google) | Open source, gratuito | ~1K stars |
| Conformal Prediction libraries | Open source, gratuito | Varios |

A maioria das ferramentas de uncertainty quantification para LLMs e open source
e gratuita, o que limita o potencial de monetizacao.

---

## 6. Sintese Final

### 6.1 Pontos Fortes

1. Codigo funcional e limpo (zero stubs, type hints, testes reais)
2. Estrutura de projeto profissional (CI/CD, packaging, pre-commit)
3. Paper academico completo com fundamentacao teorica
4. Ideia interessante (decomposicao Q1/Q2 de incerteza)
5. Infraestrutura de treinamento/avaliacao funcional
6. Licenciamento dual (permite monetizacao)

### 6.2 Pontos Fracos

1. **Resultados experimentais nao suportam as claims** (ponto mais critico)
2. Novidade tecnica limitada (~900 LOC genuinamente novel)
3. Terminologia inflacionada ("pyramidal", "fractal" nao refletem a matematica)
4. Nenhum modelo pre-treinado disponivel
5. Contradicao de licenciamento (headers proprietarios em codigo AGPL)
6. Alto percentual de codigo gerado por AI (37%)
7. Sem validacao externa ou peer review

### 6.3 Valor Justo Estimado

| Metodologia | Conservador | Moderado | Otimista |
|-------------|------------|---------|---------|
| Custo de Reposicao | US$ 21K | US$ 45K | US$ 68K |
| IP | US$ 15K | US$ 32K | US$ 50K |
| Estrategico (como componente) | US$ 30K | US$ 55K | US$ 80K |

**Ponderacao (custo 40%, IP 30%, estrategico 30%):**

| Cenario | Valor USD | Valor BRL (R$5.50) |
|---------|-----------|-------------------|
| **Conservador** | **US$ 22K** | **R$ 121K** |
| **Moderado** | **US$ 43K** | **R$ 237K** |
| **Otimista** | **US$ 65K** | **R$ 358K** |
| **Valor justo (media 30/50/20)** | **US$ 40K** | **R$ 220K** |

### 6.4 Condicoes para Aumento de Valor

| Acao | Impacto Estimado |
|------|-----------------|
| Validar experimentalmente (ECE < baseline com p < 0.05) | **+100-200%** |
| Publicar paper em conferencia peer-reviewed | +50-80% |
| Distribuir modelo pre-treinado funcional | +30-50% |
| Resolver contradicao de licenciamento | +10-20% |
| Demonstrar integracao com LLM de escala (7B+) | +50-100% |
| Obter 1+ cliente comercial | +100-300% |

### 6.5 Comparacao com ATIC

Para referencia, o aletheion-llm representa uma fracao do ecossistema ATIC:

| Metrica | aletheion-llm | ATIC (ecossistema) | Ratio |
|---------|---------------|-------------------|-------|
| LOC Python (src/) | 3,636 | 92,364 | **1:25** |
| Sistemas inovadores | 1-2 | 22+ | **1:11** |
| Papers | 1 | 6 | **1:6** |
| Teoremas formais | 0 | 11 | **0:11** |
| Benchmarks validados | 0 positivos | 5+ positivos | **0:5** |
| Valor estimado | US$ 40K | US$ 4.7M | **1:117** |

---

## 7. Nota Metodologica

Esta avaliacao foi conduzida via auditoria automatizada (contagem de linhas, leitura
de cada arquivo fonte, verificacao de outputs experimentais, analise de git history)
combinada com benchmarks de mercado para engenheiros ML e projetos open source similares.

Os valores representam estimativas e nao constituem aconselhamento financeiro formal.

---

*Documento gerado em 2026-02-28.*
