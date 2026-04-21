# Протокол оценки

Этот документ фиксирует методологические решения по оценке моделей на Yambda в рамках работы «Semantic User IDs for Recommendations». Все решения подкреплены ссылками на первоисточники и воспроизводимым benchmark-кодом авторов датасета.

Документ — authoritative: при любом расхождении с другими файлами (включая комментарии в коде) он трактуется как источник истины.

**Scope.** Протокол покрывает основные эксперименты раздела 3 плана (3.1, 3.2, 3.3, 3.5, 3.6, 3.8) и проверочные эксперименты раздела 2. Дополнительные эксперименты раздела 4 плана (обучение кодов под задачу, плоская квантизация, сегментный анализ, MLP-ранкер, CatBoost-ранкер, префиксные рекомендации, качественный анализ уровней) в scope этого протокола не входят — для них, если они выполняются, заводятся отдельные sub-protocol'ы.

## Сводная таблица решений

| # | Решение | Значение | Источник |
|---|---|---|---|
| 1 | Split scheme | Global Temporal Split, 14+14-дневные окна | Ploshkin et al. 2025 §4.1 |
| 2 | User state | A2 — frozen на момент cutoff'а | Ploshkin et al. 2025 §4.1 |
| 3 | Target set | B1 — multi-target: все positive interactions в окне | Ploshkin et al. 2025 §4.3 |
| 4 | Eval regime | Listen+, `played_ratio_pct ≥ 50` | Ploshkin et al. 2025 §4.3 |
| 5 | Scoring | Full all-item ranking, `exclude_seen = false` | Yambda benchmark `ranking.py::rank_items` |
| 6 | Метрики | Recall@K, NDCG@K, HitRate@K, Coverage@K, K ∈ {10, 100} | Järvelin & Kekäläinen 2002 |
| 7 | Training loss | Sampled softmax + log-Q correction + mixed negatives | Khrylchenko et al. 2026, Yi et al. 2018 |
| 8 | Seeds | 42, 43, 44; mean ± std | конвенция проекта |
| 9 | Mandatory baselines | MostPop + Random во всех таблицах | Cremonesi et al. 2010 |
| 10 | Significance testing | Paired Wilcoxon per-user + Bonferroni на 10 pre-specified сравнений | Sakai 2006 |

---

## 1. Split scheme: Global Temporal Split (GTS)

**Решение.** Два фиксированных timestamp-cutoff'а `T_val < T_test` делят все interactions на три непересекающихся окна: `train = [..., T_val)`, `val = [T_val, T_test)`, `test = [T_test, ...)`. Окна val и test — 14-дневные. Код: `scripts/05_global_temporal_split.py`.

**Cold-start filter.** val/test-строки с `uid` или `item_id`, отсутствующими в train, удаляются каскадом: пользователь, у которого после item-level фильтра не осталось ни одной val/test строки, выпадает из eval-популяции. Инварианты `val_users ⊆ train_users`, `val_items ⊆ train_items` проверяются в `assert_invariants`.

**Обоснование.** Ploshkin et al. 2025 §4.1 принимают GTS как стандартный protocol; leave-one-out отклонён из-за temporal dependency violation (Yambda §4.1) и memorization bias для больших моделей (ARGUS §2).

**Отклонение от canonical Yambda protocol.** Yambda paper использует 300+1 день с 30-минутным gap'ом; мы — 14+14 дней без gap'а. Обоснование 14+14: hold-out окна покрывают два полных weekly cycles и дают ~1.8M interactions для стабильной оценки; train уже содержит 87 % данных, сжатие окон ничего не купит.

Отсутствие gap'а для val-метрик создаёт потенциальный session-continuation leak (~0.15 % окна) — val numbers для session-sensitive моделей могут быть slightly inflated. Test-метрики защищены 14-дневным val-буфером. Мы **не** претендуем на bit-for-bit reproducibility Yambda Tables 6–7; цель — признанный стандартный GTS protocol.

---

## 2. User state protocol: A2 (frozen at cutoff)

**Решение.** Пользовательское представление `z_u` вычисляется **один раз** на момент cutoff'а:

- для val: `z_u_val` из истории до `T_val`;
- для test: `z_u_test` из истории до `T_test` (train ∪ val);
- для train-loop: `z_u` — causal hidden state на текущей позиции.

Внутри окна оценки `z_u` **не обновляется**. Одно `z_u` ранжирует по всему target set.

**Обоснование.** Ploshkin et al. 2025 §4.1: «All model parameters and user states were frozen at the beginning of the test period … we approximate systems with daily offline updates — a reasonable compromise for practical benchmarking.»

**Data leak ассерт.** Максимальный timestamp в prefix, из которой строится `z_u`, строго меньше `T_cutoff` соответствующего split'а.

---

## 3. Target set: B1 (multi-target в окне)

**Решение.** Для каждого val/test пользователя target = **все** его positive interactions в окне (не единичный «next item»). Пользователи с пустым target исключаются из denominator метрики.

**Обоснование.** Yambda paper §4.3 оперирует набором positive interactions в окне. Альтернативы B2 (first-item-in-window) и B3 (rolling next-item) отклонены: B2 выбрасывает 99 % окна, B3 несовместим с frozen user state.

---

## 4. Eval regime: Listen+

**Решение.** Positive feedback = listens с `played_ratio_pct ≥ 50` (integer percent). Фильтр применён в stage 3 pipeline: каждая строка `val.parquet` / `test.parquet` по построению является Listen+ positive.

**Обоснование.** Yambda paper §4.3: «To produce Listen+ we used 50 % of the track duration as the listening threshold.» Implicit feedback покрывает всю eval-популяцию, в отличие от редкого и шумного explicit Like.

---

## 5. Scoring protocol: full all-item ranking

**Решение.** Scoring по **всему** train-каталогу items (full-item, не sampled). Seen items (train, а для test — train ∪ val) **не** маскируются.

**Обоснование.** Yambda benchmark `ranking.py::rank_items` делает стандартное all-item ranking без seen-маски. Для музыкального домена re-listening — существенная часть user behavior, и Listen+ target естественно включает re-listens. Маскирование seen items скрыло бы значительную долю target и сместило бы оценку в пользу novelty-heavy моделей.

**Item matrix для scoring.**

- **SASRec-ID** (план §1.1): `I_matrix = item_encoder.embedding.weight` — параметры `nn.Embedding(n_items, d_model)` без нормализации (нормы — часть обученного представления).
- **SASRec-Content** (план §1.2): `I_matrix = L2Normalize(LinearProjection(yambda_pretrained))` — обучаемая проекция `d_item → d_model` на pretrained Yambda embeddings + row-wise L2-нормализация. Матрица предвычисляется один раз после training.

В обоих случаях `scores = z_u @ I_matrix.T`, top-K через `torch.topk` с детерминированным tie-breaking (§6). L2-нормализация в SASRec-Content стабилизирует training на pretrained embeddings с неоднородными нормами; в SASRec-ID — не применяется, потому что нормы trainable (могут кодировать popularity / confidence signal).

**Следствие.** SASRec-ID и SASRec-Content живут в **разных** пространствах item embeddings. Абсолютные Recall/NDCG между ними сравнимы (задача и target set одни), но прямое сравнение векторов (cosine similarity между embeddings) не имеет смысла без дополнительного alignment'а.

**Item id remapping.** Все модели работают с dense contiguous `item_id ∈ [0, n_items)`. Mapping `raw_uint32_id → dense_id` фиксируется на train split один раз (`artifacts/item_id_map.json`).

**Training / eval консистентность.** Training logit: `⟨h, e⟩ / exp(τ) − log Q(i)`. Eval score: просто `⟨h, e⟩`. `exp(τ)` — per-user monotonic, не меняет ranking. `log Q(i)` компенсирует sampling bias только в training; в eval sampling нет (full-catalog), bias'а нет, correction не применяется. Стандартная практика two-tower retrieval (Yi et al. 2018).

---

## 6. Метрики

**Считаются и репортятся:**

- **Recall@K** — `|top_K ∩ target| / min(K, |target|)` (capped version). Uncapped форма при `|target| > K` имеет математический максимум `K/|target|`, плохо интерпретируется. Capped всегда в `[0, 1]`. Совпадает с Yambda benchmark `metrics.py::Recall` и Yi et al. 2018.
- **NDCG@K** — `DCG@K / IDCG@K`, где `IDCG@K` по идеальному ранжированию `min(K, |target|)` элементов target set. Формула Järvelin & Kekäläinen 2002. Мы используем математически корректную версию, а не benchmark-код Yambda (у которого задокументированная variable-shadowing bug в `metrics.py::NDCG`).
- **HitRate@K** — `𝟙[top_K ∩ target ≠ ∅]`.
- **Coverage@K** — `|⋃_u R(u, K)| / |I_train|`, system-level catalog coverage.

**K values.** `K ∈ {10, 100}` — primary, согласно Ploshkin et al. 2025 Table 6.

**Агрегация.** Recall/NDCG/HitRate — mean по eval-users с непустым target в Listen+. Coverage — union по всем eval-users after cold-start filter.

**Target dedup.** Target для каждого пользователя = `Set[int]` уникальных `item_id` в окне. Повторные listen'ы одного трека входят в target как **один** элемент (IR-конвенция multi-target relevance, Järvelin & Kekäläinen 2002).

**Tie-breaking.** Deterministic: при равных scores items ранжируются по `(score DESC, item_id ASC)`. Реализация — stable two-key sort (sort по `item_id` ASC, затем stable argsort по `-score`). Арифметическая надбавка `score − ε · item_id` не используется: любая константная `ε` либо перекрывает реальные различия scores в tail-диапазоне, либо теряется в ULP float32.

**Per-user distribution (appendix).** В дополнение к mean репортятся P10, P50, P90 per-user Recall@10 и NDCG@10 — для диагностики heavy-tail артефактов.

---

## 7. Training loss: sampled softmax + log-Q + mixed negatives

**Решение.** Sampled softmax с CLIP-style learnable temperature и log-Q correction:

```
f(h_t, e_i) = ⟨h_t, e_i⟩ / exp(τ) − log Q(i)
L = − log[ exp(f(h_t, e_pos)) / (exp(f(h_t, e_pos)) + Σ_{i ∈ N} exp(f(h_t, e_i))) ]
```

где `τ` — learnable scalar, `exp(τ)` clipped в `[0.01, 100]`; `Q(i)` — оцененная частота сэмплирования negative item `i` под mixed sampler (count-min sketch).

**Negatives.** `N = N_in_batch ∪ N_uniform`:

- `N_in_batch` — positives других позиций батча с per-user boolean mask (убирает same-user positives как false negatives).
- `N_uniform` — 4096 uniform sample из train-каталога; коллизии с user-positives на Yambda-масштабе пренебрежимо маловероятны.

**Для SASRec-Content** item embeddings L2-нормализуются один раз после обучаемой проекции `d_item → d_model`. Для **SASRec-ID** нормализация не применяется (нормы — trainable degree of freedom).

**Обоснование.** Канонический паттерн sampled softmax + log-Q для large-scale retrieval (Yi et al. 2018, Khrylchenko et al. 2026 §3.1). Log-Q корректирует смещение mixed sampler: in-batch часть пропорциональна popularity (positives других пользователей концентрируются на популярных items), uniform — равномерная; деление на `Q(i)` возвращает несмещённую оценку true softmax (Bengio & Senecal 2008).

---

## 8. Seeds, агрегация, significance testing

**Seeds.** Каждый эксперимент запускается с seeds `42, 43, 44`. Результаты — `mean ± std`. Aggregation через `src/utils/aggregator.py`, читающий `saved/{run_name}/metrics.json`.

**Минимум 3 seeds** — стандарт для статистически осмысленных сравнений (с двумя нельзя посчитать std).

### Significance test

**Paired Wilcoxon signed-rank per-user** (Sakai 2006) на Recall@10 как primary метрике и NDCG@10 как secondary. Для пары моделей `(A, B)` строится массив разностей `{metric_A(u) − metric_B(u)}` по eval-users с непустым Listen+ target. Используется `scipy.stats.wilcoxon` с `alternative='two-sided'`.

**Aggregation across seeds.** `metric_A(u) = mean_{s ∈ {42, 43, 44}} metric_A(u, seed=s)` — среднее по seeds per-user. Wilcoxon работает на `N_users` точках. Альтернатива concatenation `(u, seed)` пар отвергнута — нарушает iid-assumption (3 observations с одного user коррелированы) и завышает power.

**Почему Wilcoxon.** Не требует нормальности (Recall/NDCG сильно скошены на long-tail users), мощнее paired t-test на ненормальных данных, стандарт SIGIR/TREC.

**NaN / diverged runs.** Run с loss NaN заменяется на следующий по порядку seed (45, 46, …) до 3 валидных runs. Документируется в `manifest.json` с пометкой `status: failed, step: N, reason: NaN`. Это замена невалидных runs, не cherry-picking.

**Специфика RQ-VAE.** Три seeds RQ-VAE учатся поверх **одного** reference SASRec-Content run (`sasrec_content_seed42`). Это изолирует variance quantizer'а от variance SASRec backbone; полученная variance не отражает end-to-end pipeline variance (ограничение фиксируется в `docs/risks_and_limitations.md`).

### Pre-specified сравнения

**10 pre-specified pairs**, отвечающих на research questions работы. Bonferroni α = 0.05 / 10 = **0.005**. Разница считается значимой при `p < 0.005`. При десятках тысяч eval-users мощность достаточна.

**Verification of code content (раздел 2 плана):**

1. **2.3 vs 2.4** — рекомендации по точному совпадению кода vs по перестановленному коду. Коды содержательны **именно** в смысле персонализации, а не просто удачно разбивают пользователей на группы?

**Fusion experiments (раздел 3 плана):**

2. **3.1 vs SASRec-ID baseline** — fusion с обучаемыми code-таблицами `E_d` улучшает качество поверх SASRec-ID?
3. **3.1 vs 3.1 с перестановленными кодами** — эффект от содержания кода, а не от дополнительной обучаемой ёмкости или неравномерного распределения индексов?
4. **3.1 vs 3.1 с `z_pca`** — эффект от **дискретной** структуры, а не от «любого дополнительного содержательного представления»?
5. **3.1 vs 3.2** — обучаемая интерпретация кодов (случайно инициализированные `E_d`) лучше / хуже / равна семантике RQ-VAE (фиксированные векторы из кодовых книг)? Ключевой methodological разрез работы.
6. **3.5 RQ-VAE-версия vs k-means-версия** (по схеме 3.2 с замороженной таблицей векторов) — иерархическая residual-квантизация даёт преимущество над плоской кластеризацией при **точном** равенстве обучаемых параметров и одинаковом битовом бюджете?

**Code substitution and content replacement (раздел 3 плана):**

7. **3.6 (а) настоящий код vs (в) случайный чужой код** — модель действительно чувствительна к коду пользователя, а не игнорирует его?
8. **3.8 `z_q` vs SASRec-Content baseline** — замена непрерывного вектора на код-реконструкцию сохраняет качество?
9. **3.8 `z_q` vs 3.8 с перестановленными кодами** — замена содержательна **именно для этого пользователя**?
10. **3.8 `z_q` vs PCA-реконструкция** при согласованной ошибке восстановления — дискретная квантизация лучше / хуже непрерывной линейной проекции при равном объёме потерянной информации?

**Эксперимент 3.3** (варьирование ёмкости кодов — кривые по D и K для схем 3.1 и 3.2) в список pre-specified сравнений не входит. Репортится как дескриптивная кривая «ёмкость vs качество»; интерпретация — по форме кривых (монотонность, наличие плато, сравнение форм для 3.1 и 3.2). Significance test не применяется.

### Реализация (`src/utils/stats.py`)

- `wilcoxon_paired(metric_a_per_user, metric_b_per_user) → (statistic, pvalue)` — wrapper `scipy.stats.wilcoxon`.
- `bonferroni(pvalues, alpha=0.05, n=10) → List[bool]` — деление alpha на число сравнений.
- `bootstrap_ci(metric_per_user, n_bootstrap=1000) → (lo, hi)` — 95 % CI, user-level resampling. Дополняет error bars в графиках, не заменяет significance test.

Все сравнения автоматически считаются в `aggregator.py` и попадают в `results/stats_table.csv` с колонками `(comparison, metric, k, delta_mean, pvalue, significant_bonferroni, ci_lo, ci_hi)`.

### HP tuning protocol

Гиперпараметры каждой модели (SASRec-ID, SASRec-Content, RQ-VAE, fusion-слой 3.1/3.2) подбираются **на val set** (Listen+, NDCG@10 как tuning metric). Test set используется **только** для финального отчёта, никогда для выбора HP или early stopping.

---

## 9. Mandatory baselines

В любой таблице результатов работы **обязательно** присутствуют два non-learned baseline'а — для калибровки absolute numbers.

### 9.1. MostPop

**Определение.** Одинаковый fixed top-K по всем пользователям = K самых популярных items по train-каталогу. Popularity = количество listen interactions в `data/processed/train.parquet`. Ranking — `(−pop(i), item_id)` ASC для детерминизма.

**Почему обязателен.**

1. На музыкальных streaming-datasets popularity-biased baseline даёт нетривиально высокий Recall@100 из-за Zipf-подобного распределения потребления.
2. Personalized модель, **не** обыгрывающая MostPop хотя бы на одном (K, metric), — некорректно обучена или имеет методологическую ошибку.
3. Стандарт recsys литературы (Cremonesi et al. 2010); Yambda paper включает MostPop в Tables 6–7.

**Implementation.** `scripts/train_mostpop.py` (~50 строк). Deterministic → репортится с `std = 0`.

### 9.2. Random

**Определение.** Для каждого пользователя — top-K случайно выбранных items из train-каталога (uniform без повторов). Три seeds (42, 43, 44), репортится с реальным `std`.

**Почему обязателен.** Минимальная sanity proof-of-life: learned модель, выдающая Recall в порядке Random, имеет проблемы с обучением или data leak.

### Размещение в отчётах

Все `results/*.csv` содержат строки `MostPop` и `Random` как первые две с тем же набором метрик. Это даёт ревьюеру мгновенную калибровку absolute numbers.

---

## Что сознательно не используется

- **Pairwise Accuracy (PA) / PAU** из ARGUS §4.2 — требуют production ranker и impression-level feedback, которых у нас нет.
- **Leave-last-one-out (LOO)** — memorization bias (ARGUS §2) и temporal dependency violation (Yambda §4.1).
- **Sampled-negatives metrics в eval** — full-item ranking напрямую сравним между моделями и не зависит от sample size.
- **Rolling user state update (B3)** — несовместим с A2 frozen state и сильно удорожает eval.
- **Like regime, dislikes / unlikes / undislikes** — в scope этапа 1 не входят. Работа фокусируется на implicit positive feedback (Listen+).
- **Normalized Entropy (NE)** — диагностически дублирует loss curve на одном каталоге; ranking-метрики (Recall/NDCG/HitRate) дают всю нужную информацию.

---

## References

- **Ploshkin et al. 2025** — Ploshkin, A., Tytskiy, V., et al. (2025). Yambda-5B — A Large-Scale Multi-modal Dataset for Ranking And Retrieval. *Proceedings of the 19th ACM Conference on Recommender Systems*, 894–901. arXiv:2505.22238. Локальная копия: `docs/papers/2505.22238v2.pdf`.
- **Khrylchenko et al. 2026 (ARGUS)** — Khrylchenko, K., Matveev, A., et al. (2026). Scaling Recommender Transformers to One Billion Parameters. *Proceedings of the 32nd ACM SIGKDD Conference V.1*. arXiv:2507.15994. Локальная копия: `docs/papers/argus.pdf`.
- **Yi et al. 2018** — Yi, X., Yang, J., et al. (2018). Sampling-bias-corrected neural modeling for large corpus item recommendations. *RecSys 2018*, 269–277.
- **Radford et al. 2021 (CLIP)** — Learning transferable visual models from natural language supervision. *ICML 2021*, 8748–8763. (Паттерн «dot product / `exp(τ)` с learnable temperature clipping».)
- **Bengio & Senecal 2008** — Adaptive importance sampling to accelerate training of a neural probabilistic language model. *IEEE Transactions on Neural Networks* 19(4), 713–722.
- **Järvelin & Kekäläinen 2002** — Cumulated gain-based evaluation of IR techniques. *ACM TOIS* 20(4), 422–446.
- **Cremonesi et al. 2010** — Performance of recommender algorithms on top-n recommendation tasks. *RecSys 2010*, 39–46.
- **Sakai 2006** — Evaluating evaluation metrics based on the bootstrap. *SIGIR 2006*, 525–532.
- **Yambda benchmark code** — `huggingface.co/datasets/yandex/yambda/tree/main/benchmarks`. Scoring: `ranking.py::rank_items`. Metrics: `metrics.py`.
