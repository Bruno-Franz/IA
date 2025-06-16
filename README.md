# IA
Disciplina de Inteligência Artificial

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

3. Base Textual – B2W‑Reviews01 (Hugging Face)
A base B2W‑Reviews01 disponibiliza 132 373 avaliações em português de produtos vendidos pela Americanas.com, coletadas entre janeiro e maio de 2018. O rótulo primário que empregaremos é recommend_to_a_friend (binário: Yes ou No), que indica se o consumidor recomendaria o item a terceiros.
Atributos por tipo
Texto livre principal: review_text, corpo da avaliação; precisa de tokenização e vetorização (TF‑IDF, embeddings ou similar).
Textos curtos auxiliares: review_title, product_name, product_brand. Podem ser concatenados ao texto principal ou processados separadamente.
Numéricos: overall_rating (nota de 1 a 5 estrelas) e reviewer_birth_year (ano de nascimento). A nota pode, opcionalmente, ser reagrupada em faixas de sentimento.
Categóricos: site_category_lv1, site_category_lv2, reviewer_state, reviewer_gender. Requerem codificação (one‑hot ou embeddings categóricos).
Campos de identificação ou data: submission_date, reviewer_id, product_id. Normalmente não entram no modelo, mas são úteis para filtragem ou ordenação.
Rótulo: recommend_to_a_friend, binário (Yes/No). Se se desejar testar uma tarefa multiclasse, overall_ratingpode ser transformado em sentimento negativo (1‑2), neutro (3) e positivo (4‑5).

Adequação aos Requisitos
Diversidade de formato: inclui dados tabulares, imagens e texto, cobrindo diferentes etapas de pré‑processamento e arquitetura de modelos.
Separação clara dos tipos de atributo: facilita planejar normalização, codificação ou tokenização, conforme o caso.
Tamanho gerenciável: cada conjunto cabe em equipamentos de laboratório ou notebooks pessoais, permitindo realizar as três repetições de hiperparâmetros e as cinco arquiteturas de redes neurais exigidas.
Fontes públicas confiáveis: UCI Repository, Kaggle e Hugging Face, todos com licença aberta para uso acadêmico.

https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing?utm_source=chatgpt.com
https://www.kaggle.com/datasets/alxmamaev/flowers-recognition?resource=download
https://huggingface.co/datasets/ruanchaves/b2w-reviews01

