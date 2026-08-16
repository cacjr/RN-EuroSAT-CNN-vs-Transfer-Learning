# EuroSAT — CNN vs Transfer Learning

Classificação automática de imagens de satélite em 10 categorias de cobertura terrestre, comparando uma **CNN construída do zero** com **Transfer Learning (MobileNetV2)**.

Projeto acadêmico — FIAP, curso de Inteligência Artificial.

---

## Contexto

Programas como o Copernicus (ESA/UE) e empresas da Nova Economia Espacial (Planet Labs, Airbus Defence & Space, Maxar) geram um volume de imagens de satélite grande demais para ser interpretado manualmente. Automatizar essa leitura — identificar o que cada pedaço de terra representa (floresta, lavoura, área urbana, rio, etc.) — é a base de serviços comerciais como monitoramento de desmatamento, agricultura de precisão e mapeamento de expansão urbana.

## Problema e objetivo

Dado um recorte de imagem de satélite, classificar automaticamente a que tipo de cobertura/uso do solo ele pertence, entre 10 categorias possíveis. O objetivo do projeto é comparar duas estratégias de Deep Learning para esse problema:

1. Uma **CNN construída do zero**, para entender os fundamentos arquiteturais e servir de baseline;
2. Um modelo de **Transfer Learning com MobileNetV2** pré-treinada no ImageNet, testando se o conhecimento visual genérico ajuda em um domínio tão diferente (imagens orbitais).

A pergunta central que o projeto tenta responder não é só "qual modelo classifica melhor", mas **em que condições vale a pena usar transfer learning** — e os resultados aqui mostram um caso onde a resposta é "nem sempre".

---

## Resultado principal

| Modelo | Test Accuracy | Test Loss | F1 Macro |
|---|---|---|---|
| **CNN Própria** | **90.73%** | **0.291** | **0.906** |
| MobileNetV2 (Fine-Tuned) | 87.93% | 0.567 | 0.879 |

A CNN treinada do zero superou o Transfer Learning em todas as métricas — accuracy, loss e F1 macro. Esse resultado contraria a expectativa comum (transfer learning costuma vencer) e é discutido em detalhe no notebook. As causas mais prováveis:

- **Overfitting no fine-tuning:** ao descongelar as últimas 30 camadas da MobileNetV2, a accuracy de treino passou de 99% enquanto a de validação estagnou em ~85% — sinal claro de que o modelo decorou o treino em vez de generalizar;
- **Resolução incompatível:** a MobileNetV2 foi pré-treinada em imagens 224×224; usá-la em 128×128 (resolução das imagens EuroSAT após resize) distorce parte das features herdadas do ImageNet;
- **Dataset já suficiente para uma CNN pequena:** com 10.000 imagens em um domínio visual relativamente restrito e homogêneo, uma rede simples treinada do zero consegue aprender padrões específicos do satélite sem depender de conhecimento genérico de fotos do dia a dia (ImageNet).

---

## Dataset

**EuroSAT** — dataset público de referência em sensoriamento remoto, com imagens capturadas pelo satélite Sentinel-2 (programa Copernicus/ESA), resolução espacial de 10 m/pixel.

| Característica | Detalhe |
|---|---|
| Classes | 10 (AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake) |
| Amostra usada | 1.000 imagens/classe (10.000 no total), balanceada |
| Dimensão original | 64×64 px RGB |
| Split | 70% treino / 15% validação / 15% teste (estratificado) |

## Metodologia

- Pipeline `tf.data` com cache, shuffle e prefetch
- Data Augmentation (flip, rotação, zoom, contraste) aplicado apenas no treino
- **CNN própria:** 4 blocos convolucionais (32→64→96→128 filtros) + GlobalAveragePooling + Dropout
- **MobileNetV2:** treino em 2 fases — feature extraction (base congelada) e fine-tuning (últimas 30 camadas)
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Avaliação final em conjunto de teste isolado, com matriz de confusão e classification report por classe

## Estrutura do repositório (sessão "models" aparece ao ser executado na sua máquina)

```
├── notebook/
│   └── eurosat_cnn_vs_transfer_learning.ipynb
├── models/
│   ├── cnn_eurosat_final.keras
│   └── mobilenet_eurosat_final.keras
└── README.md
```

## Como executar

1. Baixe o dataset EuroSAT ([link oficial](https://github.com/phelber/EuroSAT))
2. Abra o notebook no Google Colab (recomendado, GPU gratuita)
3. Ajuste o caminho do dataset na célula de configuração
4. Execute todas as células em ordem

## Tecnologias

`Python` · `TensorFlow / Keras` · `scikit-learn` · `pandas` · `matplotlib` / `seaborn`

## Autores

Claudio Alves · Rodrigo Wolkoff · Luna Rousseau
FIAP — Inteligência Artificial
