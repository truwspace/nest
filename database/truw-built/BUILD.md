# truw-built — Base de conhecimento semântica PT-BR para fact-checking

---

## v2 (atual, produção)

**Gerado em:** 2025-04-25
**Artigos:** 23.568 (12.252 fake + 11.076 true + 240 misleading)
**Embedding model:** paraphrase-multilingual-MiniLM-L12-v2 (384 dims)
**Formato:** .nest (SQLite + FAISS IVF+PQ + zstd)

### Arquivos v2

| Arquivo | Tamanho | Descrição |
|---|---|---|
| `truw_canonical_ptbr_v2.csv` | 48 MB | CSV canônico com 23.568 artigos e 12 colunas |
| `truw_canonical_ptbr_v2_embeddings.npy` | 35 MB | Embeddings brute float32 (23568×384) |
| — | — | .nest v2 está em `data/truw_ptbr.nest` (31 MB) |

### Schema do CSV canônico (`truw_canonical_ptbr_v2.csv`)

```
id,claim,text,source,label,label_confidence,category,date,url,language,dataset_origin,is_synthetic
```

| Campo | Descrição |
|---|---|
| `id` | ID determinístico (fb_00001, fr_00002, bt_00003, fc_00004) |
| `claim` | Título/resumo da claim (vazio se não houver) |
| `text` | Corpo completo do artigo |
| `source` | Autor ou veículo de origem |
| `label` | `fake`, `true`, ou `misleading` |
| `label_confidence` | `high` (fact-check agency), `medium` (curated corpus) |
| `category` | política, saúde, entretenimento, etc |
| `date` | Data de publicação (ISO 8601 quando disponível) |
| `url` | URL original |
| `language` | `pt-BR` |
| `dataset_origin` | `fakebr`, `fakerecogna`, `faketrue_br`, `factck_br` |
| `is_synthetic` | `false` |

### Fontes integradas (v2)

| Dataset | Artigos | Origem | Label confidence |
|---|---|---|---|
| **Fake.br-Corpus** | 7.199 | NILC/USP — pares alinhados fake/true com metadados linguísticos | medium |
| **FakeRecogna** | 11.902 | 6 agências de fact-check (boatos.org, e-farsas, etc) + veículos reais (G1, UOL) | high |
| **FakeTrue.Br** | 3.182 | GitHub CSV + HuggingFace splits | medium |
| **FACTCK.BR+** | 1.285 | TSV com claims fact-checkadas (inclui label `misleading`) | high |

**Deduplicação:** hash SHA-256 do texto normalizado (lower+strip). 3.234 duplicatas removidas. 14 artigos descartados (< 20 chars).

**NOTA:** Os datasets HuggingFace baixados (`FakeTrue.Br-hf/`, `FakeBr-hf/`, `corpus-combined/`) foram verificados e são 100% sobrepostos com os dados já em v2 — não há artigos novos a adicionar.

---

## v1 (obsoleto)

**Gerado em:** 2025-04-25 (primeira versão)
**Artigos:** 19.769 (9.874 fake + 9.895 true)

### Arquivos v1

| Arquivo | Tamanho | Descrição |
|---|---|---|
| `truw_canonical_ptbr.csv` | 43 MB | CSV canônico com 19.769 artigos e 12 colunas |
| `truw_canonical_ptbr_embeddings.npy` | 29 MB | Embeddings brute float32 (19769×384) |
| `truw_ptbr.nest` | 27 MB | Banco semântico v1 (obsoleto, não usado em produção) |

### Fontes integradas (v1)

| Dataset | Artigos | Origem | Label confidence |
|---|---|---|---|
| **Fake.br-Corpus** | 7.199 | NILC/USP — pares alinhados | medium |
| **FakeRecogna** | 11.902 | 6 agências de fact-check + veículos reais | high |
| **FakeTrue.Br** (via BiLSTM corpus) | 668 | Corpus complementar | medium |

---

## Datasets excluídos (e por quê)

| Dataset | Motivo |
|---|---|
| `News _dataset` | Duplicata byte-a-byte de `fake-truw` (ISOT, inglês) |
| `fake-truw` | ISOT dataset — inglês, notícias EUA |
| `fake_news_dataset_4000_rows` | Texto sintético genérico |
| `Nemotron-Personas-Brazil` | Personas demográficas, não notícias |
| `Brazilian-Open-Data` | Duplicata do Nemotron |
| `fakebr/` | Wrapper HuggingFace do Fake.br-Corpus — mesmo conteúdo |
| `COV19-FAKES-BR` | Removido pelo autor, indisponível |

---

## Datasets baixados mas NÃO merged (100% sobrepostos com v2)

| Diretório | Conteúdo | Notas |
|---|---|---|
| `FakeTrue.Br-hf/` | train.csv (2.865) + test.csv (717) | Subset do já merged `faketrue_br` |
| `FakeBr-hf/` | train.csv (5.760) + test.csv (1.440) | Subset do já merged `fakebr` |
| `corpus-combined/` | train.csv (8.625) + test.csv (2.157) | Merge do FakeTrue.Br + Fake.br, ambos já em v2 |

---

## Uso via nest

```python
import sys
sys.path.insert(0, "nest/src")
from nest.core import open as nest_open

db = nest_open("data/truw_ptbr.nest")

# Busca semântica por claim
results = db.search("petrobras aumentou o preço da gasolina", k=5)
for r in results:
    print(f"[{r.label.upper()}] score={r.score:.4f} src={r.source}")
    print(f"  {r.text[:120]}...")
```

---

## Como reconstruir (v2)

```bash
# FASE 1: ETL (CSV canônico v2)
python3 scripts/etl_canonical_v2.py  # → truw_canonical_ptbr_v2.csv

# FASE 2: Embeddings
python3 -c "
  from sentence_transformers import SentenceTransformer
  import csv, numpy as np
  texts = [r['claim']+'\n'+r['text'] if r['claim'] else r['text']
           for r in csv.DictReader(open('dataset/truw-built/truw_canonical_ptbr_v2.csv'))]
  model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
  emb = model.encode(texts, normalize_embeddings=True, batch_size=256).astype(np.float32)
  np.save('dataset/truw-built/truw_canonical_ptbr_v2_embeddings.npy', emb)
"

# FASE 3: Build .nest
nest build dataset/truw-built/truw_canonical_ptbr_v2.csv \
  -e dataset/truw-built/truw_canonical_ptbr_v2_embeddings.npy \
  -o data/truw_ptbr.nest \
  -m paraphrase-multilingual-MiniLM-L12-v2
```

---

## Validação com claims reais

Queries do log do engine (124 claims brasileiras) testadas no .nest:

| Query | Top-1 label | Score | Relevância |
|---|---|---|---|
| `petrobras aumentou o preço da gasolina` | FAKE | 0.64 | Direto — nota fiscal da Bolívia |
| `vacina covid causa morte` | TRUE | 0.43 | Artigo sobre vacinas, baixa confiança |
| `bolsonaro foi preso` | TRUE | 0.59 | Prisões políticas relacionadas |
| `haddad aumentou imposto de renda` | FAKE | 0.75 | Aumento de combustíveis (análogo) |

A busca semântica resolve o problema de sinonímia identificado no log ("prender" ≈ "detido"), substituindo jaccard de shingles por cosine similarity em embeddings multilíngues.