# IA
Disciplina de Inteligência Artificial

[Resultados dos experimentos](RESULTS.md)

Bases de Dados Selecionadas
Para cumprir o requisito de trabalhar com três formatos distintos (tabular, imagem e texto) e avaliar diferentes técnicas de classificação supervisionada, foram escolhidos conjuntos públicos de tamanho médio, rótulos bem definidos e licença aberta para uso acadêmico. 

1. Base Tabular – Bank Marketing (UCI / Kaggle)
O conjunto Bank Marketing reúne 45 211 registros de chamadas de telemarketing feitas por um banco português entre 2008 e 2010. Cada instância descreve um cliente contactado, e o objetivo é prever se ele subscreve (yes) ou não (no) um depósito a prazo. A base está em CSV e não contém valores ausentes.
Atributos por tipo
Numéricos contínuos ou de contagem: idade (age), saldo médio anual (balance), duração da chamada em segundos (duration), dia do mês da chamada (day), número de contatos na campanha atual (campaign), dias desde o último contato anterior (pdays), total de contatos em campanhas anteriores (previous). Estes campos podem exigir padronização antes do treinamento.
Categóricos: profissão (job), estado civil (marital), nível de educação (education), tipo de contato (contact – telefone fixo ou celular), mês da chamada (month) e resultado da campanha anterior (poutcome). Todos precisam ser codificados (por exemplo, one‑hot).
Binários: histórico de inadimplência (default), existência de empréstimo habitacional (housing) e existência de empréstimo pessoal (loan). Já vêm como “yes”/“no” e podem ser convertidos diretamente em 0/1.
Rótulo: campo y, binário, indicando a adesão ao produto financeiro.

2. Base de Imagens – Flowers Recognition (Kaggle)
O dataset Flowers Recognition contém 4 242 fotografias JPEG (~320 × 240 px), divididas em cinco pastas que correspondem às espécies daisy, dandelion, rose, sunflower e tulip. O problema de interesse é classificar cada imagem na sua categoria correta.
Atributos por tipo
Matriz de pixels RGB: cada imagem representa uma amostra; é recomendável redimensionar para um tamanho uniforme (p.ex. 224 × 224 px) e normalizar valores de cor antes de alimentar uma CNN.
Rótulo: a espécie da flor, derivada diretamente da pasta que contém a imagem (cinco classes no total).

3. Base Textual – Books Reviews em Português (GitHub)
O conjunto Books Reviews reúne 2 000 avaliações de livros publicadas por usuários da Amazon Brasil. Metade dos comentários está no arquivo `books_pt_neg` e foi classificada como negativa (abaixo de 3 estrelas); a outra metade encontra-se em `books_pt_pos` e corresponde a resenhas positivas (acima de 3 estrelas). O problema é identificar automaticamente se o texto expressa opinião favorável ou desfavorável.
Atributos por tipo
Texto livre principal: cada linha de texto contém uma resenha completa, que precisa ser tokenizada e vetorizada (TF‑IDF ou embeddings) para alimentar os modelos.
Rótulo: 0 para `books_pt_neg` e 1 para `books_pt_pos`. Como os arquivos não trazem metadados adicionais, o foco é exclusivamente na classificação textual.

Adequação aos Requisitos
Diversidade de formato: inclui dados tabulares, imagens e texto, cobrindo diferentes etapas de pré‑processamento e arquitetura de modelos.
Separação clara dos tipos de atributo: facilita planejar normalização, codificação ou tokenização, conforme o caso.
Tamanho gerenciável: cada conjunto cabe em equipamentos de laboratório ou notebooks pessoais, permitindo realizar as três repetições de hiperparâmetros e as cinco arquiteturas de redes neurais exigidas.
Fontes públicas confiáveis: UCI Repository, Kaggle e Hugging Face, todos com licença aberta para uso acadêmico.

https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing?utm_source=chatgpt.com
https://www.kaggle.com/datasets/alxmamaev/flowers-recognition?resource=download
https://github.com/larifeliciana/books-reviews-portuguese

## Configuração do ambiente

1. Use Python 3.10 ou superior. É recomendável criar um ambiente virtual:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Instale as dependências principais:

   ```bash
   pip install pandas scikit-learn tensorflow tensorflow-datasets matplotlib seaborn
   ```

## Download dos conjuntos de dados

- **Bank Marketing** – faça o download de `bank.zip` no site da UCI e extraia `bank-full.csv`:

  ```bash
  wget https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip
  unzip bank.zip
  ```

- **Flowers Recognition** – o dataset é obtido automaticamente quando `tfds.load("tf_flowers")` é executado.
- **Books Reviews** – o arquivo `books_reviews.csv` já acompanha este repositório (a versão original está em [books-reviews-portuguese](https://github.com/larifeliciana/books-reviews-portuguese)).


## Execução do script

Execute todo o fluxo de experimentos com:

```bash
python main.py
```

O script carrega as três bases, aplica os modelos principais e em seguida executa os baselines de árvore de decisão definidos em `baseline_dt.py`. Todas as métricas são exibidas no terminal.

Para gerar um CSV consolidado com essas métricas, utilize o módulo `baseline_dt` diretamente:

```python
import pandas as pd
import baseline_dt

pd.DataFrame(baseline_dt.run_all()).to_csv('resultados_consolidados.csv', index=False)
```

O arquivo `Análise do trabalho prático.pdf` apresenta a interpretação dos resultados obtidos. Recrie esse documento sempre que atualizar o CSV.

