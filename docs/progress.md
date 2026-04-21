# Прогресс реализации

Документ отражает текущий статус работ. Шаги привязаны к `docs/experiment_plan_revised.md` и `docs/eval_protocol.md`. Отмечаются в `[x]`.

## Сводный статус

- [x] **Этап 0** — eval harness + обязательные baseline'ы
- [x] **Этап 1** — SASRec-ID и SASRec-Content (§1.1, §1.2 плана)
- [ ] **Этап 2** — RQ-VAE + проверочные эксперименты (§2 плана)
- [ ] **Этап 3** — основные эксперименты (§3.1–§3.8 плана)
- [ ] **Этап 4** — дополнительные эксперименты (§4 плана)
- [ ] Написание текста диплома

---

## Этап 0: eval harness + MostPop + Random

### 0.1 Инфраструктура утилит `src/utils/`
- [x] `seed.py` — `set_seed`, `resolve_device`, `free_device_memory`
- [x] `io.py` — `save_metrics`, `load_metrics`, `save_npz`, `load_npz`
- [x] `stats.py` — `wilcoxon_paired`, `bonferroni`, `bootstrap_ci`
- [x] `aggregator.py` — `aggregate_seeds`, `write_results_csv`

### 0.2 Item ID remapping + data reader
- [x] `scripts/09_build_item_id_map.py` → `artifacts/item_id_map.json` (260 927 items)
- [x] `src/data/dataset.py` — `ItemIdMap`, `build_targets`, `load_user_ids`, `load_popularity`
- [x] `src/data/paths.py` — абсолютные пути + `resolve_split_parquet` (full / 10pct / 1pct)

### 0.3 Ranking infrastructure (порт `yambda/evaluation/ranking.py`, Apache 2.0)
- [x] `src/metrics/ranking.py` — `Embeddings`, `Targets`, `Ranked`, `rank_items`
- [x] Детерминированный tie-breaking через stable descending argsort (item_id ASC при равных scores)
- [x] `Targets.__post_init__` делает per-user dedup (B1 multi-target set semantics)

### 0.4 Metrics (порт `yambda/evaluation/metrics.py` с правками)
- [x] `src/metrics/metrics.py` — `Recall`, `Precision`, `DCG`, `MRR`, `Coverage`
- [x] **Фикс NDCG**: используется `ideal_target_mask` в знаменателе (yambda-баг устранён)
- [x] **Добавлен `HitRate`** (в yambda отсутствует)
- [x] `per_user_primary` — per-user Recall@k и NDCG@k для будущих Wilcoxon

### 0.5 Mandatory baselines `src/baselines/`
- [x] `popularity.py` — `rank_mostpop` (top-K популярных с tie-break item_id ASC)
- [x] `random_rec.py` — `rank_random` (uniform top-K без повторов на seed)

### 0.6 Hydra конфиги `configs/`
- [x] `data/yambda.yaml`, `trainer/default.yaml`, `wandb/default.yaml`
- [x] `mostpop.yaml`, `random.yaml`

### 0.7 Entry scripts `scripts/`
- [x] `train_mostpop.py`, `train_random.py`, `aggregate_baselines.py`

### 0.8 Unit-тесты `tests/`
- [x] `test_metrics.py` — NDCG ideal=1, HitRate, Recall capping, target dedup, DCG, per-user primary
- [x] `test_ranking.py` — `rank_items` vs brute-force, tie-breaking, Embeddings sort invariant
- [x] `test_baselines.py` — MostPop одинаковый top-K, MostPop >> Random, Random ≈ K/n_items

### 0.9 Финальные результаты этапа 0
- [x] `results/baselines.csv`
- [x] 16/16 unit-тестов зелёные
- [x] MostPop test/Recall@100 = 0.067, Random = 0.0007 → 100× разрыв (sanity ok)

---

## Этап 1: SASRec-ID и SASRec-Content

### 1.0 Скачать референс yambda
- [x] `benchmarks/models/sasrec/{model,train,data,eval}.py` через WebFetch

### 1.1 `src/sasrec/model.py`
- [x] `SASRec` с параметром `item_source ∈ {trainable, pretrained}`
- [x] Reverse positional encoding (newest → pos 0) сохранён
- [x] Shift dense→model_id (`+1`) внутри item_encoder; padding_idx=0 не конфликтует с item 0
- [x] `encode_full_history` — no-grad last-token extraction для A2 state
- [x] `item_matrix()` — L2-нормирован для Content, сырой для ID (eval_protocol §5)
- [x] `enable_nested_tensor=False` в TransformerEncoder — обход MPS

### 1.2 `src/sasrec/dataset.py` + `collate.py`
- [x] `TrainSequenceDataset` — causal next-item (last `max_seq_len`)
- [x] `EvalStateDataset` — A2 prefix с фильтром по user_ids
- [x] `load_prefix_sequences` — assert data-leak (max timestamp в prefix < T_cutoff)
- [x] `load_temporal_cutoffs` читает `data/splits_metadata.json`

### 1.3 `src/sasrec/loss.py`
- [x] `SampledSoftmaxLoss` — learnable `τ` с clipping `exp(τ) ∈ [0.01, 100]`
- [x] Mixed negatives: in-batch (per-user mask) + uniform `n_uniform=4096`
- [x] Closed-form log-Q: `q_uniform = 1/n_items`, `q_inbatch ∝ popularity`
- [x] `item_encoder` хранится в list-обёртке — без дублирования параметров

### 1.4 `src/sasrec/trainer.py` + `extract.py` + `eval_loop.py`
- [x] Adam, `grad_clip_norm=1.0`, val early stopping (patience=3)
- [x] Best checkpoint → `model_best.pth`
- [x] `EvalContext` кэш — prefix sequences и targets строятся один раз
- [x] Ranking/metrics выполняются на CPU (обход MPS OOM/fragmentation)
- [x] `extract_and_save` — `z_u` для train/val/test + `item_matrix.npy`
- [x] `free_device_memory` + `gc.collect()` между evalами

### 1.5 Hydra configs + entry script
- [x] `configs/model/sasrec.yaml`, `configs/trainer/sasrec.yaml`
- [x] `configs/sasrec_id.yaml`, `configs/sasrec_content.yaml`
- [x] `scripts/train_sasrec.py` — единый entry point через `--config-name`

### 1.6 Unit-тесты SASRec
- [x] `tests/test_sasrec.py` — 13 тестов:
  - shape'ы forward; causal mask blocks future; padding invariance
  - `encode_full_history` = last-token; eval-mode детерминизм
  - L2 нормы item_matrix для Content vs ID
  - sampled softmax finite/backward/log-Q; отсутствие дублирования параметров

### 1.7 Scale testing
- [x] Smoke на `subsample_1pct` — инфраструктура работает
- [x] Scale на `subsample_10pct` — SASRec-ID val/Recall@10 ≈ 0.138 > MostPop 0.086 (1.60×)
- [x] Исправлен MPS OOM через CPU-eval + cached EvalContext + `empty_cache`

### 1.8 Final runs (full × 3 seeds × 2 модели)
- [x] `sasrec_id_seed{42,43,44}` (hyperparams: d_model=64, max_seq_len=128, batch=64, 15 epochs)
- [x] `sasrec_content_seed{42,43,44}`
- [x] Артефакты: `saved/{run}/`, `artifacts/user_vectors/{run}/`

### 1.9 Aggregate
- [x] `scripts/aggregate_sasrec.py`
- [x] `results/sasrec_id.csv`, `results/sasrec_content.csv`, `results/sasrec_summary.csv`

### 1.10 Итог этапа 1 (test, full)

| Model | Recall@10 | NDCG@10 | vs MostPop |
|---|---|---|---|
| SASRec-ID | 0.143 ± 0.001 | 0.166 ± 0.0004 | **1.89×** |
| SASRec-Content | 0.058 ± 0.0008 | 0.061 ± 0.001 | 0.77× (ниже MostPop) |

- [x] 52/52 тестов зелёные (после двух проходов code review этапа 1)
- [x] SASRec-ID > MostPop (требование плана выполнено)
- [ ] **Открыто:** SASRec-Content < MostPop. После CR этапа 1 найден баг в `PretrainedItemEncoder.forward` (отсутствовала L2-нормализация — несоответствие `eval_protocol §5/§7`); исправлено в коде. Артефакты `artifacts/user_vectors/sasrec_content_seed{42,43,44}/` обучены с багом и подлежат **перегенерации перед этапом 2** (вход RQ-VAE). Только после переобучения можно судить, остаётся ли underperformance содержательным ограничением аудио-эмбеддингов.

---

## Этап 2: RQ-VAE + проверочные эксперименты (§2 плана)

### 2.0 Инфраструктура
- [ ] `src/rqvae/model.py` — `RQVAE` с D уровнями квантизации, K векторов в каждом codebook
- [ ] `src/rqvae/codebook.py` — отдельная VQ-codebook с EMA-обновлениями + dead code reset
- [ ] `src/rqvae/loss.py` — reconstruction + commitment losses
- [ ] `src/rqvae/trainer.py` — обучение на `z_u^{содерж.}` из `artifacts/user_vectors/sasrec_content_seed42/`
- [ ] `src/rqvae/extract_codes.py` — получение `c = (c_1, …, c_D)` per user → `artifacts/user_codes/`
- [ ] Hydra configs: `configs/model/rqvae.yaml`, `configs/rqvae.yaml`
- [ ] Unit-тесты: VQ-lookup, EMA update, dead-code reset, reconstruction gradient flow

### 2.1 Обучение RQ-VAE (§2.1 плана)
- [ ] Стартовая конфигурация: D=4, K=256 (32 бита на пользователя)
- [ ] Log codebook utilization per level в W&B — assert > 50%
- [ ] Seeds 42/43/44 поверх `sasrec_content_seed42` (одна referent SASRec-backbone, см. eval_protocol §8)
- [ ] Артефакт: `artifacts/user_codes/D4_K256_sasrec_content_seed42/{train,val,test}_codes.npy`

### 2.2 Подбор D и K (§2.2 плана)
- [ ] D ∈ {2, 4, 6, 8} при K=256
- [ ] K ∈ {32, 64, 256, 1024} при D=4
- [ ] Дополнительно D=4, K=32 (20 бит) — для контроля с k-means в §3.5
- [ ] `results/rqvae_ablation.csv` с reconstruction error + codebook utilization
- [ ] Выбор рабочей (D, K) — фиксируется до §3

### 2.3 Рекомендации по точному совпадению кодов (§2.3)
- [ ] `src/retrieval/code_retrieval.py` — full match → top-K популярных в группе
- [ ] Все вычисления — без обучения, только поиск по таблицам
- [ ] `results/retrieval_fullmatch.csv`

### 2.4 Рекомендации по перестановленным кодам (§2.4)
- [ ] Random permutation истинных кодов между пользователями
- [ ] Сохраняет маргинальное распределение кодов, разрушает связь «пользователь — код»
- [ ] `results/retrieval_permuted.csv`

### 2.5 k-NN и popularity baselines (§2.5)
- [ ] k-NN в пространстве `z_u^{содерж.}` (cosine)
- [ ] Popularity (уже из этапа 0, но параллельные таблицы ожидаются здесь)
- [ ] Ожидаемый порядок: k-NN ≥ full_match > permuted ≥ popularity

---

## Этап 3: основные эксперименты (§3 плана)

### 3.1 (ОСН) Fusion с обучаемыми code-таблицами (§3.1)
- [ ] Extend SASRec-ID: concatenate `z_u^{ID}` с суммой обучаемых `E_d[c_d]`
- [ ] Обучаются только линейный слой объединения + таблицы `E_d`
- [ ] Контроль: permuted codes
- [ ] Контроль: `z_{pca}` (10–15 компонент)
- [ ] Wilcoxon pair #2, #3, #4 из eval_protocol §8

### 3.2 (ОСН) Fusion с фиксированными RQ-VAE векторами (§3.2)
- [ ] `z_q = Σ codebook_d[c_d]` как frozen вектор
- [ ] Обучается только линейный слой объединения
- [ ] Wilcoxon pair #5 (3.1 vs 3.2)

### 3.3 (ОСН) Варьирование ёмкости кода (§3.3)
- [ ] Повторить 3.1 и 3.2 для всех (D, K) из 2.2
- [ ] Две параллельные кривые качество vs ёмкость (дескриптивный результат)

### 3.4 (ДОП) Вклад уровней иерархии (§3.4)
- [ ] Обучение с random level masking; при eval — только первые k уровней
- [ ] Кривая качество vs глубина используемого кода

### 3.5 (ОСН) Контроль с k-means (§3.5)
- [ ] MiniBatchKMeans с `2^bits` кластеров на `z_u^{содерж.}`
- [ ] Сравнение RQ-VAE-vs-kmeans по схеме 3.2 (frozen таблицы)
- [ ] Wilcoxon pair #6

### 3.6 (ОСН) Подмена кода (§3.6)
- [ ] На уже обученной 3.1 модели — 3 варианта кода (настоящий, соседа, случайный)
- [ ] Wilcoxon pair #7

### 3.7 (ДОП) Рекомендации только по семантическому коду (§3.7)
- [ ] Матричная факторизация `z_u = Σ E_d[c_d]`, `score = z_u · v_i`
- [ ] Без трансформера, без истории в явном виде
- [ ] Оценка: превосходит ли popularity, приближается ли к SASRec-ID

### 3.8 (ОСН) Замена непрерывного вектора на `z_q` (§3.8)
- [ ] На уже обученной SASRec-Content заменить `z_u^{содерж.}` на `z_q` при eval
- [ ] Контроли: permuted codes, PCA-реконструкция (подобран `k` под reconstruction error RQ-VAE)
- [ ] Wilcoxon pairs #8, #9, #10

### 3.9 Aggregate
- [ ] `results/exp3_*.csv` и `results/stats_table.csv` (10 pre-specified сравнений с Bonferroni)

---

## Этап 4: дополнительные эксперименты (§4 плана)

В порядке приоритета:

- [ ] **4.1** Обучение кодов с учётом задачи (совместный loss RQ-VAE + recsys)
- [ ] **4.2** Плоская квантизация при равном битовом бюджете
- [ ] **4.3** Анализ качества по квартилям длины истории
- [ ] **4.4** MLP-ранкер (5 конфигураций представления пользователя)
- [ ] **4.5** CatBoost-ранкер (семантические ID как категориальные признаки)
- [ ] **4.6** Рекомендации по префиксу кода (k=1…D−1)
- [ ] **4.7** Качественный анализ уровней (стабильность по префиксам истории, однородность акустики)

---

## Артефакты на текущий момент

```
artifacts/
├── item_id_map.json                              [0.2]
└── user_vectors/
    ├── sasrec_id_seed{42,43,44}/                 [1.8]
    └── sasrec_content_seed{42,43,44}/            [1.8]

saved/
├── mostpop/                                      [0.9]
├── random_seed{42,43,44}/                        [0.9]
├── sasrec_id_seed{42,43,44}/                     [1.8]
└── sasrec_content_seed{42,43,44}/                [1.8]

results/
├── baselines.csv                                 [0.9]
├── sasrec_id.csv                                 [1.9]
├── sasrec_content.csv                            [1.9]
└── sasrec_summary.csv                            [1.9]
```

**Тесты:** 52/52 зелёные (`uv run pytest tests/`).

---

## Code review этапа 0 (post-mortem)

Внешний субагент-ревьювер прошёлся по `src/utils/`, `src/data/`, `src/metrics/`, `src/baselines/`, `configs/`, `scripts/` и `tests/`. Все исправленные находки:

- **MAJOR.** `src/utils/aggregator.py`: `df[col].std(ddof=0)` (population std) → заменено на `ddof=1` с обработкой `NaN → 0.0` для одного seed (MostPop). Все `results/*.csv` пересчитаны: std-колонки выросли в `√(3/2) ≈ 1.225` раз для агрегатов по 3 seeds.
- **MAJOR.** `src/metrics/ranking.py::Targets.from_sequential`: метод не делал `group_by("uid")`, ассерт уникальности падал бы на любых реальных данных. Добавлен `group_by + agg + sort`. В Stage 0/1 не вызывался — но теперь готов к Stage 2.
- **MINOR.** `src/baselines/popularity.py`: убран лишний каст `popularity → float32` (counts > 2²⁴ теряли бы точность). Сортировка теперь идёт прямо в исходном `int64`; в `scores` каст в `float32` уже после выбора top-k.
- **MINOR.** `src/metrics/__init__.py`: добавлен экспорт `per_user_primary` (использовался через прямой импорт).
- **MINOR.** `tests/`: добавлены 8 тестов на edge cases — `k = n_items`, `k > n_items` для `rank_mostpop`/`rank_random`/`rank_items`, регрессия для `Targets.from_sequential` с дублированием `uid`, проверка int64-точности в `rank_mostpop`.

## Code review этапа 1 (post-mortem)

Внешний субагент-ревьювер прошёлся по `src/sasrec/` (model, dataset, collate, loss, trainer, eval_loop, extract), `scripts/train_sasrec.py`, `scripts/aggregate_sasrec.py`, `configs/sasrec_*`, `tests/test_sasrec.py`. Ложные тревоги (in-batch diagonal masking, data-leak assert, loss-buffer load) ревьювер сам отозвал. Реально подтверждённые правки:

- **CRITICAL.** `src/sasrec/model.py::PretrainedItemEncoder.forward` возвращал `self.proj(audio_emb)` без L2-нормализации, тогда как `item_matrix` нормализует — train (sampled-softmax) и eval (`z_u @ I_matrix.T`) использовали разные пространства. Несоответствие `eval_protocol §5` (Training/eval консистентность) и `§7` («Для SASRec-Content item embeddings L2-нормализуются один раз после проекции»). Добавлен `F.normalize(...)` в `forward`. Это объясняет (хотя бы частично) underperformance SASRec-Content относительно MostPop. Артефакты `z_u^{content}` подлежат регенерации.
- **MINOR.** Добавлен тест `test_pretrained_forward_matches_item_matrix_rows`: для одного и того же dense_id `forward(id)` ↔ `item_matrix()[id]` совпадают, обе единичной нормы — регрессия для C1.
- **MINOR.** Добавлен тест `test_sampled_softmax_inbatch_diagonal_masked`: для batch с двумя позициями × двумя пользователями диагональ и same-user пары в `inbatch_logits` маскируются `-inf`.
- **MINOR.** Добавлен тест `test_load_prefix_sequences_rejects_data_leak`: на мок-parquet с `timestamp ≥ T_val` функция бросает `AssertionError` с сообщением «Data leak».

## Code review этапа 1 (второй проход)

Ещё один независимый проход субагента-ревьювера по тем же файлам. CRITICAL не обнаружено. Исправленные находки:

- **MAJOR.** `src/sasrec/dataset.py::load_user_sequences` опирался на недокументированное поведение Polars: `df.sort("timestamp") → group_by("uid") → agg(pl.col("item_id"))` — внешняя сортировка не гарантируется при `group_by + agg(list)`, порядок был корректен только де-факто в Polars 1.39.3. Заменено на `group_by + agg(pl.col("item_id").sort_by("timestamp"))` — гарантия порядка выражена явно внутри агрегирующего выражения. Docstring обновлён.
- **MAJOR.** `src/sasrec/trainer.py::fit`: после финального `_run_eval()` добавлен `assert (save_dir / "model_best.pth").exists()`. Ранее `_load_checkpoint` тихо логировал warning при отсутствии файла (возможно при пустом eval_loader или NaN в tuning-metric), и `extract_and_save` работал бы на состоянии последней эпохи вместо лучшего чекпоинта.
- **MINOR.** Добавлен тест `test_load_user_sequences_respects_timestamp_order`: parquet с нарочно перепутанными по времени строками подтверждает, что `load_user_sequences` возвращает sequences в хронологическом порядке — регрессия для MAJOR#1.
