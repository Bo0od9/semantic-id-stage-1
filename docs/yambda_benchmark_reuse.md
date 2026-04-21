# Переиспользование benchmark-кода Yambda

Документ фиксирует, какие части официального benchmark-кода Yambda стоит взять за основу при реализации моделей, какие — адаптировать, а какие — пропустить. Цель — не переписывать с нуля то, что уже проверено авторами датасета, и при этом не унаследовать их методологические решения, расходящиеся с нашим eval-протоколом.

## Где живёт код

Весь benchmark Yambda находится **внутри HuggingFace dataset repo**, а не на GitHub (`github.com/yandex/yambda` и `github.com/yandex-research/yambda` — 404).

- Корень: https://huggingface.co/datasets/yandex/yambda/tree/main/benchmarks
- Лицензия: Apache 2.0
- Автор: @ploshkin (первый автор Yambda paper)

Структура:
```
benchmarks/
├── yambda/
│   ├── constants.py              — 587 B
│   ├── utils.py                  — 1.05 kB
│   ├── processing/
│   │   ├── chunk_read.py         — 2.22 kB
│   │   └── timesplit.py          — 7.44 kB
│   └── evaluation/
│       ├── metrics.py            — 7.14 kB
│       └── ranking.py            — 5.20 kB
├── models/
│   ├── popularity/main.py        — 5.47 kB
│   ├── random_rec/main.py        — 3.49 kB
│   ├── itemknn/main.py           — 8.06 kB
│   ├── bpr_als/main.py           — 9.45 kB
│   ├── sansa/main.py             — 8.67 kB
│   └── sasrec/
│       ├── model.py              — 11.9 kB
│       ├── train.py              — 4.61 kB
│       ├── data.py               — 6.81 kB
│       └── eval.py               — 4.23 kB
└── scripts/
    ├── get_dataset_stats.py      — 7.09 kB
    ├── make_multievent.py        — 2.85 kB
    └── transform2sequential.py   — 2.33 kB
```

---

## Что берём (high-value)

### 1. `evaluation/ranking.py` → `src/metrics/ranking.py`

**Брать почти один-в-один.** Это ядро retrieval-инфраструктуры.

Что там есть:
- `@dataclass Embeddings` — пара `(ids: Tensor, embeddings: Tensor)` с проверками сортировки и уникальности; методы `save(file_path)` / `load(file_path)` через `np.savez`. Подходит и для хранения `z_u`, и для item-эмбеддингов.
- `@dataclass Targets` — `user_ids: Tensor` + `item_ids: list[Tensor]`. Это **ровно формат multi-target (B1)** из `eval_protocol.md` item 3: каждому пользователю — список positive items в окне оценки. Есть `Targets.from_sequential(df, device)` — конструктор из polars/parquet.
- `@dataclass Ranked` — результат ранжирования: `user_ids`, `item_ids`, `scores`; асёрты на сортировку scores по убыванию.
- `rank_items(users, items, num_items, batch_size=128)` — полный all-item dot product + `topk`, батчами по user. **Без exclude-seen.** Ровно item 5 нашего протокола («Full all-item ranking, exclude_seen = false»).

Правок почти не требуется. Возможно, пересмотреть hardcoded `batch_size=128` в сторону параметра.

### 2. `evaluation/metrics.py` → `src/metrics/metrics.py` — **с обязательными правками**

**Интерфейс красивый, но есть баги.** Брать структуру, но метрики пересчитать.

Что там есть:
- Абстрактный класс `Metric` + реализации: `Recall`, `Precision`, `NDCG`, `DCG`, `MRR`, `Coverage`.
- `calc_metrics(ranked, targets, metrics: list[str])` — принимает строки вида `"ndcg@100"`, парсит через `_parse_metrics`, возвращает `dict[name][k] = float`.
- `create_target_mask(ranked, targets)` — строит `(n_users, max_k)` маску «попал ли i-тый top-K item в target-множество пользователя».

#### Баги, которые нужно починить

**Баг 1: NDCG использует не ту маску для идеального ранжирования.**

В коде:
```python
ideal_target_mask = (
    torch.arange(...)[None, :] < targets.lengths[:, None]
).to(torch.float32)

assert target_mask.shape == ideal_target_mask.shape

ideal_dcg = calc_dcg(target_mask)   # ← BUG: должно быть calc_dcg(ideal_target_mask)
```

`ideal_target_mask` вычислен, но не использован. В результате `ideal_dcg == actual_dcg`, и NDCG даёт 1 везде, где `actual_dcg ≠ 0` — т.е. фактически меряется не NDCG.

**Починить:** заменить `calc_dcg(target_mask)` → `calc_dcg(ideal_target_mask)` в вычислении `ideal_dcg`.

**Баг 2: неверная агрегация NDCG по пользователям.**

Комментарий в коде: _«we computed (dcg_1 + ... + dcg_n) / (idcg_1 + ... + idcg_n) instead of (1 / n) * (dcg_1 / idcg_1 + ... + dcg_n / idcg_n)»_.

После фикса бага 1 проверить, что `divide()` реально делает `(x/y).mean()` с нулём там, где `y == 0`. Стандартная формула NDCG — это среднее по пользователям от `dcg_u / idcg_u`.

**Баг 3: Recall cap — это не баг.**

В коде `torch.clamp(num_positives, max=k)` знаменатель у Recall@K. Комментарий называет это багом, но для multi-target это **правильное** поведение: если у пользователя 5 positives, а K=10, знаменатель = 5; если positives=50, знаменатель = 10. Оставить как есть.

#### Чего не хватает

`HitRate@K` (item 6 нашего eval-протокола) не реализован. Добавить:
```python
class HitRate(Metric):
    def __call__(self, ranked, targets, target_mask, ks):
        return {k: (target_mask[:, :k].sum(dim=-1) > 0).float().mean().item() for k in ks}
```

Также в `REGISTERED_METRIC_FN` дописать `"hitrate": HitRate()`.

### 3. `models/sasrec/model.py` → `src/sasrec/model.py`

**Брать `SASRecEncoder` почти один-в-один; BCE loss из `forward()` выкинуть.**

Детально разобрано в предыдущем обсуждении. Ключевое:
- Построен на `nn.TransformerEncoder` с `batch_first=True`
- Causal mask (нижнетреугольная) + padding mask
- Позиционные эмбеддинги в **обратном** порядке (newest = position 0) — **важная деталь**, которая отличается от классических SASRec-имплементаций
- Trunc-normal init с `initializer_range=0.02`
- `num_embeddings = num_items + 1` с зарезервированным 0 под padding
- В eval-режиме возвращает last-token embedding → это и есть `z_u` под протокол A2

Правки:
- Убрать из `forward()` ветку training с BCE loss — выносится в `src/sasrec/loss.py` (sampled softmax + log-Q correction + mixed negatives по ARGUS §3.1)
- `forward()` всегда возвращает `(batch_size, seq_len, d_model)` + mask; loss считается снаружи
- Добавить метод `encode_full_history(items) -> z_u` для построения `z_u` по полной истории пользователя до cutoff (протокол A2)

**Дефолты из их `train.py`** (ориентир по размеру):
- `embedding_dim=64`, `num_heads=2`, `num_layers=2`, `max_seq_len=200`
- `batch_size=256`, `lr=1e-3` (Adam), `dropout=0.0`, `num_epochs=100`

`max_seq_len=200` для наших данных мал (медиана истории 1727, p90 ≈ 7.8K) — расширить до 512, при памяти и возможности до 1024.

### 4. `models/sasrec/eval.py` → паттерн для `src/sasrec/extract_user_vectors.py`

**Не копировать буквально, но использовать как шаблон.**

Показывает, как SASRec встраивается в `rank_items()`-пайплайн:
```python
def infer_users(eval_dataloader, model, device):
    # проход по EvalDataset, сборка (user_ids, user_embeddings)

def infer_items(model):
    return model.item_embeddings.weight.data

# далее:
item_embeddings = Embeddings(ids=arange(n_items), embeddings=item_emb_matrix)
user_embeddings = Embeddings(ids=user_ids, embeddings=user_emb)
ranked = rank_items(users=user_embeddings, items=item_embeddings, num_items=100)
metrics = calc_metrics(ranked, targets, metrics=metric_names)
```

Правки под наш проект:
- `num_items=100` — хардкод в их коде; заменить на `max(K_values)`
- Добавить сохранение `z_u_val` и `z_u_test` в `artifacts/user_vectors/{exp_name}/` для downstream использования в RQ-VAE и MLP-ранкере (протокол A2 требует **один** `z_u` на cutoff, не rolling)
- Для SASRec-Content: `infer_items` возвращает не `model.item_embeddings.weight.data`, а предвычисленные аудио-эмбеддинги (проекция не нужна, если eval делается в пространстве `d_model`)

---

## Брать как reference, переписать под свои данные

### 5. `models/popularity/main.py` → `src/baselines/popularity.py`

Time-decayed popularity с hyperopt по полураспаду:
```python
tau = decay ** (1 / DAY_SECONDS / (hour / 24))
item_score = sum(tau ** (max_timestamp - timestamp)) по всем прослушиваниям item-а
```

Что берём:
- Функцию `training(hour, train, max_timestamp, device, decay=0.9)` — ~20 строк чистой логики
- Идею hyperopt по `hours ∈ {0, 0.5, 1, 2, 6, 12, 24, 168, ...}` на val, финал на test

Что меняем:
- Data pipeline (у них `timesplit.flat_split`) — заменить на наши `data/processed/{train,val,test}.parquet`
- Добавить вариант без time-decay (plain popularity) — он у них получается как `hour=0`

Это более качественный вариант baseline-а 2.5 из плана, чем просто «топ K».

### 6. `models/random_rec/main.py` → `src/baselines/random_rec.py`

Тривиально. Нужен только для item 10 eval-протокола («MostPop + Random во всех таблицах»). Усреднение по `num_repeats=2` прогонам.

Взять только паттерн интеграции в `Ranked` → `calc_metrics` (~15 строк). Остальное — свой код.

### 7. `yambda/utils.py` → `src/utils/`

1 kB, две полезных функции:
- `mean_dicts(list_of_dicts)` — нужно для агрегации метрик по сидам (mean ± std по 42/43/44)
- `argmax(list, key=...)` — для hyperopt

В `src/utils/aggregator.py` проверить, есть ли аналоги; если нет — перенести.

---

## Пропускаем

| Файл | Причина |
|---|---|
| `yambda/__init__.py`, `evaluation/__init__.py`, `processing/__init__.py` | пустые |
| `yambda/constants.py` | их конкретные значения (`LAST_TIMESTAMP=26000000`, `GAP_SIZE=1800`, `TEST_SIZE=DAY_SECONDS`) специфичны для full-Yambda; у нас свои cutoffs из stage 5. Список `METRICS` взять можно, но проще захардкодить свой |
| `yambda/processing/timesplit.py` | наш GTS split уже зафиксирован в stage 5 со своими cutoffs (14+14 дней, без 30-min gap); их `flat_split` / `sequential_split` не совместимы с нашими processed parquets |
| `yambda/processing/chunk_read.py` | chunked parquet нужен для multi-TB raw Yambda; наш processed датасет влезает в RAM |
| `models/itemknn/main.py` | item-item CF с time-decay; не фигурирует в нашем плане. Наш k-NN baseline (план 2.5) — по **user-векторам** `z_u^содерж.`, это другой смысл |
| `models/bpr_als/main.py` | BPR-MF / iALS — классические recsys baselines, но план их не требует |
| `models/sansa/main.py` | sparse autoencoder, не в плане |
| `models/sasrec/train.py` | BCE loss с одним негативом (наш — sampled softmax + log-Q + mixed negatives); нет validation-loop; click-CLI (у нас Hydra); data pipeline несовместим. Проще написать свой trainer |
| `models/sasrec/data.py` | использует polars + их `timesplit`; у нас уже готовые parquets со своим split |
| `scripts/*.py` | работают на raw Yambda; у нас свои stages 01–08 |

---

## Критические различия в методологии

Если брать их код, нужно помнить, где оригинал расходится с нашим eval-протоколом:

| Аспект | Yambda benchmark | Наш протокол | Где решать |
|---|---|---|---|
| Loss | BCE с одним случайным негативом | sampled softmax + log-Q correction + mixed negatives | `src/sasrec/loss.py` пишем свой |
| Validation during training | нет | early-stopping по val Recall@10 | `src/sasrec/trainer.py` пишем свой |
| Split | 300d train / 30-min gap / 1d test | 272d train / 0 gap / 14d val / 14d test (stage 5) | уже сделано |
| `z_u` protocol | last-token от всей истории | A2: frozen at cutoff (`T_val`, `T_test`) | в `extract_user_vectors` |
| NDCG корректность | 2 бага (см. выше) | fixed | при копировании `metrics.py` |
| HitRate@K | не реализован | нужен по плану | дописать класс |
| `num_ranked_items` | 100 хардкод | `max(K ∈ {10, 100, 20, 50}) = 100` | совпадает, но параметризовать |
| Config | click CLI | Hydra | свои `configs/` |

---

## Чек-лист для начала реализации

**Порядок работы:**

1. [ ] Создать `src/metrics/ranking.py` — скопировать `Embeddings`, `Targets`, `Ranked`, `rank_items` из `evaluation/ranking.py`. Добавить unit-тест: проверить, что `rank_items` даёт те же top-K, что brute-force на маленьком примере.

2. [ ] Создать `src/metrics/metrics.py` — скопировать `Metric`, `Recall`, `Precision`, `NDCG`, `DCG`, `MRR`, `Coverage`, `calc_metrics`, `create_target_mask`. **Починить NDCG** (баги 1 и 2). **Дописать `HitRate`**. Unit-тест на игрушечном примере с известным ответом для NDCG и HitRate.

3. [ ] Создать `src/baselines/popularity.py` — адаптировать `training()` из их `popularity/main.py`. Hyperopt по `hours` на val.

4. [ ] Создать `src/baselines/random_rec.py` — тривиально, по паттерну их `random_rec/main.py`.

5. [ ] Создать `src/sasrec/model.py` — скопировать `SASRecEncoder` из их `model.py`, убрать BCE loss из `forward()`, добавить `encode_full_history()` метод.

6. [ ] Создать `src/sasrec/loss.py` — sampled softmax + log-Q correction + mixed negatives (ARGUS §3.1). **Своё, их не брать.**

7. [ ] Создать `src/sasrec/trainer.py` — training loop с val early-stopping. **Своё.**

8. [ ] Создать `src/sasrec/dataset.py` — `TrainDataset` / `EvalDataset` поверх наших `data/processed/*.parquet`. Паттерн collate_fn взять из их `data.py` (variable-length → packed tensor + lengths — аккуратная схема).

9. [ ] Создать `src/sasrec/extract_user_vectors.py` — по паттерну их `eval.py::infer_users`, но с сохранением в `artifacts/user_vectors/{exp_name}_{seed}/`.

10. [ ] Сделать SASRec-Content вариант — наследник `SASRecEncoder`, `_item_embeddings` заменён на `nn.Embedding.from_pretrained(audio_embeddings, freeze=True)` + линейная проекция до `d_model`, если `audio_dim != d_model`.

**Проверки после реализации:**

- [ ] Recall@10 на Random-baseline: совпадает с ожидаемым `k / n_items` в пределах шума
- [ ] Recall@100 на Popularity: разумно выше Random (типично 5-20× для recsys)
- [ ] NDCG@10 на SASRec-ID после починки багов: сравнимо с их paper Table 6/7 numbers (±50%, с учётом всех отклонений от их setup — split, loss, data subset)
- [ ] Codebook utilization > 50% для RQ-VAE на всех уровнях (требование CLAUDE.md)

---

## Ссылки для копирования кода

Прямые URL на `resolve/main/` для удобного `wget` / `curl`:

```
https://huggingface.co/datasets/yandex/yambda/resolve/main/benchmarks/yambda/evaluation/ranking.py
https://huggingface.co/datasets/yandex/yambda/resolve/main/benchmarks/yambda/evaluation/metrics.py
https://huggingface.co/datasets/yandex/yambda/resolve/main/benchmarks/yambda/utils.py
https://huggingface.co/datasets/yandex/yambda/resolve/main/benchmarks/models/sasrec/model.py
https://huggingface.co/datasets/yandex/yambda/resolve/main/benchmarks/models/sasrec/eval.py
https://huggingface.co/datasets/yandex/yambda/resolve/main/benchmarks/models/sasrec/data.py
https://huggingface.co/datasets/yandex/yambda/resolve/main/benchmarks/models/popularity/main.py
https://huggingface.co/datasets/yandex/yambda/resolve/main/benchmarks/models/random_rec/main.py
```
