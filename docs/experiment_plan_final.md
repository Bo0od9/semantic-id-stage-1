# План экспериментов: Semantic User IDs for Recommendations

## Что это за работа

Цель — проверить, **улучшают ли hierarchical semantic user IDs качество рекомендаций** в задаче sequential next-item recommendation (предсказание следующего трека, который пользователь захочет послушать, по его истории прослушиваний).

**Semantic user ID** — это короткая последовательность дискретных кодов `(c_1, c_2, c_3, c_4)`, каждый из которых — целое число от 0 до K−1 (у нас K=256). Вместо того, чтобы описывать пользователя плотным вектором из 64+ чисел с плавающей точкой, мы описываем его четырьмя целыми числами. Идея: такой компактный «ID» может оказаться полезнее, чем continuous вектор, потому что он структурирован (верхние уровни кодируют общие предпочтения, нижние — тонкие нюансы).

**Три research question (RQ):**

- **RQ1 — сохранение качества.** Можно ли заменить continuous user-вектор на 4 целых числа без значимой потери качества?
- **RQ2 — добавление сигнала.** Добавляют ли коды полезную информацию **поверх** continuous представления?
- **RQ3 — структура.** Важна ли именно иерархия кодов или достаточно flat квантизации того же битового бюджета?

**Где тестируется каждый RQ:**
- RQ1 → §4.1 Replace.
- RQ2 → §4.3 Fusion-B (главный эксперимент всей работы).
- RQ3 → §5.1 Hierarchical vs flat.

## Данные

**Yambda dataset** — музыкальный датасет Яндекса с миллионами user-track interactions плюс предобученные аудио-эмбеддинги треков (128-мерные вектора, отражающие акустическое содержание трека).

**Eval protocol** — Global Temporal Split (два cutoff'а делят данные по времени на train / val / test, val и test — 14-дневные окна в конце). Cold-start users / items в val и test исключаются. Все метрики считаются на Listen+ positives (треки, прослушанные ≥ 50% длительности).

**Метрики** — Recall@K, NDCG@K, HitRate@K, Coverage@K. Primary `K ∈ {10, 100}`. Full all-item ranking. Все эксперименты — 3 seeds (42, 43, 44), результаты как mean ± std. Значимость сравнений — paired Wilcoxon signed-rank per-user с Bonferroni correction.

Полные детали — в `docs/eval_protocol.md`. План ниже cross-references этот документ для методологических вопросов.

---

## 1. Базовые модели

В работе используются **две разные SASRec модели** с разными ролями. SASRec — это transformer-based архитектура для sequential recommendation: на вход получает последовательность items, которые пользователь слушал, на выход выдаёт вектор `z_u` — компактное представление пользователя.

### 1.1 SASRec-ID: стандартный SASRec с learnable item embeddings

**Как устроен.** Классический SASRec. Каждому треку в каталоге соответствует trainable embedding — вектор из обучаемых параметров (начинается случайным, на обучении подстраивается). На вход модели подаётся последовательность `embedding(track_1), embedding(track_2), ...` — история пользователя. Transformer по этой последовательности выдаёт `z_u^{ID}` — вектор, который описывает пользователя через то, **какие треки он слушал вместе** (collaborative pattern).

**Как обучается.** На train split с sampled softmax + log-Q correction (детали — `eval_protocol.md` §7). Задача — предсказать следующий трек по предыдущей истории. Оптимизируются item embeddings и веса transformer'а одновременно.

**Как используется в плане.**
- **Baseline** в таблицах: фиксирует верхнюю границу качества classic SASRec.
- **§4.3 Fusion-B**: главный downstream эксперимент — проверяем, помогают ли semantic codes этой модели.

**Зачем нужен.** Это «collaborative» SASRec — он знает про паттерны совместного потребления треков, но ничего не знает про их аудио-содержание напрямую. Идеальная среда для теста «добавляют ли content-based коды сигнал поверх collaborative signal?».

### 1.2 SASRec-Content: SASRec с аудио-эмбеддингами

**Как устроен.** Отличается от SASRec-ID только входом. Вместо обучаемых `embedding(track_i)` подаются **фиксированные аудио-эмбеддинги** из Yambda (CNN, обученная Яндексом на спектрограммах треков). Эти эмбеддинги не обучаются — они frozen. Поверх них применяется небольшая trainable linear projection `d_audio → d_model`, чтобы адаптировать под SASRec encoder.

Выход — `z_u^{content}` — вектор, описывающий пользователя через то, **какие аудио-характеристики треков он предпочитает** (content-based taste).

**Как обучается.** Так же, как SASRec-ID (тот же loss, тот же train split). Отличие в том, что обучается меньше параметров: только projection layer и transformer weights, но не item embeddings (они frozen).

**Как используется в плане.**
- **Второй baseline** в таблицах — рядом с SASRec-ID.
- **Источник `z_u` для §2 (RQ-VAE).** RQ-VAE будет квантовать именно `z_u^{content}`, потому что нам нужны коды, описывающие **вкус** пользователя (содержательные предпочтения), а не collaborative patterns.
- **§4.1 Replace**: внутри SASRec-Content проверяем, сохраняют ли коды качество при прямой замене `z_u` на реконструкцию из кодов.

**Зачем нужен.** Чтобы коды описывали вкус через реальное аудио-содержание треков. Например, если у двух пользователей одинаковый `c_1 = 12` на верхнем уровне иерархии, то это должно означать, что они слушают треки с похожими акустическими характеристиками (например, оба любят рок).

### 1.3 PCA compact baseline

**Как устроен.** Берутся `z_u^{content}` всех train-пользователей из §1.2. На этой матрице обучается PCA (стандартный метод линейного сжатия размерности). Каждый пользователь проецируется в пространство меньшей размерности — получается `z_{pca}`. Размерность выбирается так, чтобы cumulative explained variance была разумной (около 50%, практически это ≈10–15 компонент; см. ограничение R-005 в `risks_and_limitations.md`).

**Как используется.** Как контроль в §4 и §6. Если semantic codes работают лучше PCA при похожем «бюджете» на сжатие — значит, полезна именно дискретная иерархическая структура, а не просто компактность.

**Зачем нужен.** Без этого контроля невозможно отделить выигрыш от discretization как таковой от выигрыша от «любого compressed representation».

---

## 2. Построение semantic user IDs

### 2.1 RQ-VAE (reconstruction-only)

**Что такое RQ-VAE.** Residual Quantized Variational Autoencoder — метод, который берёт dense вектор и превращает его в последовательность дискретных кодов через жадную иерархическую квантизацию.

**Как работает.** У RQ-VAE есть D уровней, на каждом — codebook из K векторов (небольшая таблица). Дано — вектор `z_u`. На первом уровне ищем в codebook №1 ближайший вектор к `z_u`, запоминаем его индекс `c_1` (число от 0 до K−1). Вычитаем этот вектор из `z_u` — получаем residual. На втором уровне ищем ближайший вектор в codebook №2 уже к этому residual'у, индекс `c_2`. И так далее. В итоге каждый пользователь описывается D числами `(c_1, c_2, ..., c_D)` — его semantic user ID.

**Что обучается.** Сами codebooks — таблицы из K × d_model чисел, одна на уровень. Обучаются так, чтобы сумма выбранных codebook-векторов `z_q = Σ codebook_d[c_d]` была как можно ближе к исходному `z_u`. Loss = reconstruction (`‖z_u − z_q‖²`) + commitment loss (не даёт codebook-векторам уплывать слишком далеко).

**Как используется в плане.** RQ-VAE обучается на `z_u^{content}` из §1.2 (всех train-пользователей). Стартовая конфигурация: **D=4 уровня, K=256 кодов на уровень** — то есть 32 бита на пользователя (4 × log₂ 256 = 32). Меры против codebook collapse (ситуация, когда часть кодов никогда не выбирается): EMA-обновления + reset мёртвых кодов.

**Что мониторим.** Codebook utilization (доля используемых кодов на каждом уровне — должна быть > 50%) и reconstruction error (насколько хорошо `z_q` приближает `z_u`).

**Зачем нужен этап.** Получить working semantic user IDs и убедиться, что дискретизация технически возможна. Это prerequisites для всех дальнейших экспериментов — если codebook collapse'нулся или reconstruction error catastrophic, нужно сначала починить это.

### 2.2 Ablation по глубине D и размеру codebook K

**Что делаю.** Повторяю §2.1 с разными конфигурациями:
- D ∈ {2, 4, 6, 8} при K=256;
- K ∈ {64, 256, 1024} при D=4.

Для каждой конфигурации фиксирую reconstruction error, codebook utilization и downstream quality через §3.1 (retrieval sanity).

**Зачем нужен.** Найти оптимальную конфигурацию (D, K) — она будет использоваться во всех главных экспериментах §4. Если не проверить заранее, есть риск получить негативный результат только из-за неудачного D или K.

**Как выбираем.** Оптимальная конфигурация — та, у которой best tradeoff между reconstruction error (ниже — лучше) и downstream retrieval quality (выше — лучше). Фиксируется **до** запуска §4 и не меняется впоследствии.

---

## 3. Семантическая осмысленность кодов (sanity)

Этот блок **не отвечает на главный вопрос** об improvement. Его задача — убедиться, что коды вообще несут осмысленную информацию о пользователях. Если этот блок провалится, результаты §4 интерпретировать нельзя. Блок лёгкий — никакого обучения, только lookup по готовым артефактам.

### 3.1 Full-match retrieval по кодам

**Что делаю.** Для каждого test пользователя: (1) беру его код `(c_1, ..., c_D)`, вычисленный на prefix истории; (2) ищу всех train-пользователей с **точно таким же кодом**; (3) рекомендую top-K самых популярных items внутри этой группы.

**Зачем нужен.** Самый чистый тест на «содержательность» кодов. Если пользователи с одинаковым кодом имеют похожие вкусы — популярные items среди них должны быть хорошей рекомендацией. Никакого обучения, только hash lookup.

### 3.2 Random codes — контроль

**Что делаю.** То же, что §3.1, но каждому пользователю присваивается **случайный** код той же структуры (D уровней, K значений). Retrieval идёт так же, но по random bucket'ам.

**Зачем нужен.** Убедиться, что §3.1 работает **именно потому, что коды семантичны**, а не потому, что «любое разбиение пользователей на группы работает». Если §3.1 и §3.2 дают одинаковый результат — RQ-VAE коды не лучше случайных, и интерпретация §4 как «тест semantic IDs» теряет смысл.

### 3.3 k-NN и popularity — baselines

**Что делаю.**
- **k-NN:** для каждого test пользователя нахожу top-k ближайших train-пользователей по cosine similarity в пространстве `z_u^{content}`, рекомендую популярные items у соседей.
- **Popularity (MostPop):** глобальный top-K самых популярных items в train, без персонализации.

**Зачем нужны.** (a) k-NN показывает «целевой» уровень — насколько хорошо работает similarity в **continuous** пространстве. Если discrete full-match (§3.1) близок к k-NN — коды эффективно сохраняют similarity структуру. (b) MostPop — нижняя граница. Любая personalized модель должна его обыгрывать.

**Ожидаемый порядок метрик.** k-NN ≥ full-match (§3.1) > random codes (§3.2) ≈ MostPop. Если такой порядок подтверждается — коды семантичны, можно идти к §4.

---

## 4. Главные эксперименты: улучшают ли коды качество рекомендаций?

Это ядро работы. Прямые тесты RQ1 и RQ2.

### 4.1 Replace: z_u → z_q внутри SASRec-Content (RQ1)

**Где эксперимент.** Внутри SASRec-Content из §1.2.

**Что делаю.** У обученного SASRec-Content при scoring заменяю `z_u^{content}` (оригинальный user-вектор) на `z_q^{content}` (реконструкция из codebook: `z_q = Σ codebook_d[c_d]`). То есть вместо полного continuous user-вектора модель получает вектор, восстановленный из 4 целых чисел. SASRec-Content не переобучается — только inference-time замена. Оцениваю метрики на test.

**Контроли.**
- **Replace с random codes** (те же random codes, что §3.2, реконструкция `z_q^{rand}` из случайных индексов). Ожидается, что качество упадёт до уровня MostPop — это подтверждает, что выигрыш §4.1 (если он есть) — от семантики, а не от формы представления.
- **Replace с PCA-вектором** (`z_{pca}` из §1.3). Если `z_q` ≈ `z_{pca}` по качеству — дискретная структура не добавляет пользы относительно простого сжатия.

**Зачем нужен.** Прямой тест RQ1: «сохраняют ли 32 бита кодов качество continuous представления?»

**Circularity примечание.** Да, `z_q` — это реконструкция того же `z_u^{content}`, поэтому эксперимент формально «circular» (коды и вектор — из одного источника). Но это **не проблема**, потому что RQ1 именно этот вопрос и задаёт: «можно ли заменить z_u его reconstruction'ом из кодов?». Мы проверяем сохранение информации, не её добавление.

**Интерпретация.**

| Результат | Вывод |
|---|---|
| Replace ≥ 95% от SASRec-Content baseline | Сильное сохранение качества: дискретизация «практически бесплатна» |
| Replace 80–95% | Partial loss: коды сохраняют большую часть сигнала |
| Replace < 80% | Существенная потеря: нужно пересмотреть D, K |
| Replace с random ≈ MostPop | Подтверждает, что успех (если есть) — от семантики RQ-VAE |
| Replace ≈ Replace с PCA | Дискретная структура сама по себе не даёт выигрыша, только сжатие |

### 4.2 Fusion-A: [z_u^{ID}; z_q^{content}] внутри SASRec-ID (RQ2, frozen decoding)

**Где эксперимент.** Внутри **SASRec-ID** — параллельно §4.3, но с другим способом декодирования кодов.

**Что делаю.** При scoring SASRec-ID использую concatenation `[z_u^{ID}; z_q^{content}]` + trainable linear projection `h = W[z_u^{ID}; z_q^{content}] + b`. Где:
- `z_u^{ID}` — user-вектор из SASRec-ID (collaborative pattern).
- `z_q^{content} = Σ codebook_d[c_d]` — реконструкция из **frozen** codebook RQ-VAE (содержит content-based taste).

Projection `W` обучается на train; SASRec-ID body и RQ-VAE codebook frozen.

**Важный момент — non-circularity.** `z_u^{ID}` и `z_q^{content}` — из **разных моделей** и живут в **разных пространствах**. `z_u^{ID}` не имеет прямого доступа к аудио-содержанию треков (он знает только collaborative patterns). `z_q^{content}` несёт content-based taste (реконструкция `z_u^{content}`). Следовательно, fusion может реально добавить информации — это **не circular setup**.

**Зачем нужен.** Прямой тест: «добавляет ли **frozen RQ-VAE reconstruction** сигнал к collaborative SASRec-ID?»

**Прямое сравнение с §4.3 Fusion-B.** Оба эксперимента — в SASRec-ID, оба используют коды `(c_1, ..., c_D)` из RQ-VAE над SASRec-Content. Отличие — **только в способе декодирования кодов в dense vector для fusion**:
- **Fusion-A (здесь):** `z_q = Σ codebook^{RQ-VAE}[c_d]` — готовые вектора из RQ-VAE, frozen.
- **Fusion-B (§4.3):** `code_emb = Σ E_d[c_d]` — новые обучаемые таблицы, init random, trained под recommendation task.

Сравнение §4.2 vs §4.3 изолирует вопрос: «достаточно ли frozen RQ-VAE reconstruction, или коды нужно re-learn под задачу?»

**Теоретическое замечание.** Альтернативный вариант «Fusion-A внутри SASRec-Content» (то есть `[z_u^{content}; z_q^{content}]` в одном и том же пространстве) не запускается как отдельный эксперимент, потому что он **предсказуемо даст null gain**: `z_q` — это `z_u` с небольшой reconstruction error, fusion вектора с самим собой не добавляет информации. Это математически очевидный факт, не требующий эмпирической проверки.

**Контроли.** Те же, что §4.3 (оба эксперимента используют одни и те же контрольные группы):
- Fusion-A с random codes.
- Fusion-A с `z_{pca}` вместо `z_q^{content}`.

### 4.3 Fusion-B: [z_u^{ID}; learnable code embeddings] внутри SASRec-ID (RQ2, главный эксперимент)

**Где эксперимент.** Внутри **SASRec-ID** (не SASRec-Content!). Это принципиально, см. «Зачем именно SASRec-ID» ниже.

**Что делаю.** При scoring SASRec-ID использую concatenation `[z_u^{ID}; code_emb]`, где

```
E_d = nn.Embedding(K, d_user_shared)   # D таблиц, trainable, init random
code_emb = Σ_{d=1..D} E_d[c_d]
h = W[z_u^{ID}; code_emb] + b          # fusion projection
```

**Важное отличие от §4.2.** §4.2 и §4.3 — **тот же setup** (внутри SASRec-ID, тот же `z_u^{ID}`, те же коды из RQ-VAE над SASRec-Content), различаются только способом декодирования кодов в dense vector:
- **§4.2 Fusion-A:** `z_q = Σ codebook^{RQ-VAE}[c_d]` — **готовые** (frozen) вектора из RQ-VAE codebook.
- **§4.3 Fusion-B (здесь):** `code_emb = Σ E_d[c_d]` — **новые обучаемые** таблицы `E_d`, init random, обучаемые с нуля под recommendation task.

В §4.3 коды используются как чистые **integer индексы** — метки категории, смысл которых выучивается под задачу. Каждая строка `E_d[k]` постепенно превращается в вектор, оптимальный для recommendation, за счёт gradient updates (Adam подкручивает только те строки, чьи коды встретились в batch'е).

Обучается: fusion projection `W` + таблицы `E_d`. SASRec-ID body frozen.

**Зачем именно SASRec-ID (а не SASRec-Content).** Это нужно, чтобы избежать **circularity**:

- Коды получены из `z_u^{content}` (SASRec-Content + RQ-VAE). Они кодируют **content-based taste** пользователя.
- `z_u^{ID}` (из SASRec-ID) кодирует **collaborative pattern** — совместное потребление треков. Он напрямую не знает про аудио-характеристики.
- Это **две разных модальности информации**. Коды могут нести сигнал, которого в `z_u^{ID}` нет.

Если бы мы делали §4.3 внутри SASRec-Content, `z_u^{content}` и коды были бы из одного источника, и теоретический потолок выигрыша был бы около нуля (`I(codes; target) ≤ I(z_u^{content}; target)`).

**Контроли.** Все — внутри SASRec-ID:
- **Fusion-B с random codes** (random коды той же структуры, те же learnable E_d, trained the same way). Отделяет «коды несут сигнал» от «learnable capacity сама по себе помогает».
- **Fusion-B с flat-VQ codes** — коды из flat VQ того же битового бюджета (см. §5.1 ниже). Отделяет вклад иерархии от вклада любой дискретизации.
- **Fusion с `z_{pca}`** (проекция `z_u^{content}` через PCA). Показывает, не даёт ли простое continuous compression тот же эффект, что discrete codes. Если Fusion-B ≈ Fusion-PCA — выигрыш от supplementary representation, не от discretization.

**Зачем нужен.** **Главный тест RQ2.** Прямой вопрос: «добавляют ли content-based semantic codes сигнал поверх collaborative SASRec?»

**Интерпретация.** Ключевые сравнения для RQ2:

| Результат | Вывод |
|---|---|
| Fusion-B > SASRec-ID baseline **и** Fusion-B > Fusion-B-random | Коды несут task-relevant сигнал, complementary к collaborative SASRec — **главный положительный ответ на RQ2** |
| Fusion-B > SASRec-ID, но ≈ Fusion-B-random | Выигрыш — от learnable capacity (любые обучаемые параметры помогают), не от семантики кодов |
| Fusion-B ≈ SASRec-ID | Коды не добавляют сигнала поверх collaborative — RQ2 отвергается |
| Fusion-B > Fusion-PCA | Discrete structure полезнее continuous compression — сильное подтверждение RQ2 |
| Fusion-B ≈ Fusion-PCA | Benefit — от наличия supplementary representation в целом, а не от discretization |
| Fusion-B > Fusion-A (§4.2) | Learnable интерпретация кодов эффективнее, чем frozen RQ-VAE decoding — важный методологический результат |
| Fusion-B ≈ Fusion-A | Frozen RQ-VAE reconstruction уже достаточна — learnable таблицы не нужны |

---

## 5. Глубина иерархии и rec-aware обучение (RQ3)

Эти эксперименты отвечают на оставшиеся RQ из proposal. Запускаются после §4 и используют его выводы.

### 5.1 Hierarchical vs flat при равном бюджете бит (RQ3)

**Что делаю.** Обучаю **flat VQ** — один codebook, без residual quantization. Flat VQ кодирует пользователя **одним** индексом из большого codebook'а (например, K=65536, что даёт 16 бит на пользователя). Сравниваю с hierarchical RQ-VAE того же битового бюджета (например, D=2, K=256 — тоже 16 бит). Повторяю §4.1 Replace и §4.3 Fusion-B с flat codes.

**Зачем нужен.** Прямой тест RQ3: «does hierarchical residual quantization offer advantages over flat?»

**Интерпретация.**
- Hierarchical Fusion-B > flat Fusion-B: иерархия существенна.
- Сопоставимо: достаточно любой дискретизации того же бюджета, hierarchical структура не несёт отдельного value.

### 5.2 Rec-aware objectives

**Что делаю.** Дообучаю RQ-VAE совместно с recommendation head: loss = λ_rec × reconstruction + λ_bpr × (sampled softmax на next-item prediction). Получаю новые коды, обученные под recommendation task. Повторяю §4.1 (Replace) и §4.3 (Fusion-B) с этими кодами.

**Зачем нужен.** Третий RQ proposal: «does recommendation-aware learning of user codes make such identifiers more beneficial?»

**Интерпретация.**
- Rec-aware Fusion-B > reconstruction-only Fusion-B: коды нужно оптимизировать под задачу.
- Сопоставимо: reconstruction objective достаточен.

---

## 6. Дополнительные эксперименты (при наличии времени)

Эти эксперименты **не отвечают на главный вопрос** об improvement в sequential recommendation. Они покрывают смежные вопросы (transferability кодов, интерпретация уровней иерархии, subgroup-анализ). Полезны для более полной картины в дипломе, но не блокируют финальный вывод.

### 6.1 Cross-model transfer: MLP ранкер

**Что делаю.** Обучаю простой MLP scorer (`Linear → ReLU → Linear → ReLU → Linear → scalar`) для pointwise scoring на (user_repr, item_emb). Item embeddings — Yambda pretrained, frozen. 5 configurations user representation: (a) zeros (без персонализации), (b) learnable from scratch (обычный learnable user embedding), (c) frozen `z_u^{content}` (continuous из SASRec-Content), (d) frozen `z_{pca}` (PCA-сжатый), (e) semantic IDs через learnable per-level embeddings (как в §4.3 Fusion-B, но в рамках MLP). Все configs разделяют MLP body; user-side приводится к общей размерности через projection (детали — `eval_protocol.md` §10).

**Зачем нужен.** Проверить **transferability** кодов: работают ли они в другой модели, не связанной с SASRec. Это **не вопрос improvement**, а вопрос «можно ли переиспользовать коды в других downstream задачах».

### 6.2 Cross-model transfer: CatBoost ранкер

**Что делаю.** То же, что §6.1, но на CatBoost (градиентный бустинг, принципиально другой класс моделей). Semantic codes подаются как категориальные фичи — CatBoost их нативно поддерживает.

**Зачем нужен.** Проверить, что transferability не специфична для нейросетей.

### 6.3 Prefix match retrieval

**Что делаю.** Для каждого `k ∈ {1, ..., D−1}`: matching по первым `k` уровням кода (менее специфичный match, больше группа train-пользователей). Рекомендации — top-K популярных items внутри группы. Fallback к `k−1` при группе размера < 5; трекается `fallback_rate@k`.

> **⚠️ REVISIT перед запуском.** Protocol fallback-к-`k−1` оставлен с пометкой пересмотреть. Альтернативы: (a) no fallback → sparse users получают MostPop; (b) metric@k + share_non_unique@k репортятся раздельно; (c) гибрид. Проблема текущего protocol: heterogeneous effective k per user confound'ит monotonicity interpretation. Финальное решение — после §2.1, когда известно фактическое distribution group sizes.

**Зачем нужен.** Проверить, кодируют ли верхние уровни кодов широкие предпочтения, а нижние — узкие нюансы (как заявлено в proposal).

### 6.4 Subgroup analysis

**Что делаю.** Разбиваю пользователей на квартили по длине истории. Для §4.1, §4.3, §6.1 считаю метрики по группам отдельно.

**Зачем нужен.** Гипотеза: relative выигрыш от semantic codes больше на пользователях с короткой историей (prior от кода компенсирует недостаток данных). Проверить её.

### 6.5 Анализ семантики уровней иерархии

**Что делаю.** (a) Для каждого пользователя вычисляю коды по разным prefix'ам истории (70%, 80%, 90%, 100%), считаю долю совпадений кода на каждом уровне. (b) Группирую пользователей по коду первого уровня, проверяю корреляцию с жанровыми предпочтениями (если есть жанровая разметка — иначе через ближайших соседей в item embeddings).

**Зачем нужен.** Проверить hypothesis: верхние уровни кодов стабильны (кодируют long-term taste), нижние — меняются с историей (short-term context).

---

## Сводка экспериментов

### Главные эксперименты (отвечают на главный вопрос)

| # | Эксперимент | Где | RQ | Зачем |
|---|---|---|---|---|
| 1.1 | SASRec-ID | — | — | Collaborative baseline |
| 1.2 | SASRec-Content | — | — | Content baseline + источник z_u для §2 |
| 1.3 | PCA baseline | — | — | Control для compression effect |
| 2.1 | RQ-VAE | на z_u^content | — | Построить semantic IDs |
| 2.2 | Ablation D, K | на z_u^content | — | Оптимальная конфигурация |
| 3.1 | Full-match retrieval | — | — | Sanity: коды содержательны |
| 3.2 | Random codes | — | — | Sanity: не артефакт |
| 3.3 | k-NN + popularity | — | — | Sanity: calibration |
| **4.1** | **Replace** | **SASRec-Content** | **RQ1** | **Сохраняют ли коды качество при прямой замене?** |
| 4.2 | Fusion-A (frozen decoding) | SASRec-ID | RQ2 | Помогает ли frozen RQ-VAE reconstruction collaborative SASRec? |
| **4.3** | **Fusion-B (learnable decoding)** | **SASRec-ID** | **RQ2** | **Главный эксперимент: помогают ли коды как обучаемые tokens?** |

### Secondary core (оставшиеся RQ)

| # | Эксперимент | RQ | Зачем |
|---|---|---|---|
| 5.1 | Hierarchical vs flat | RQ3 | Иерархия vs flat VQ при равном бюджете |
| 5.2 | Rec-aware objectives | RQ3 | Нужно ли учить коды под recommendation task |

### Дополнительные (не отвечают на главный вопрос)

| # | Эксперимент | Зачем |
|---|---|---|
| 6.1 | MLP transfer | Transferability кодов |
| 6.2 | CatBoost transfer | Устойчивость к классу модели |
| 6.3 | Prefix match | Интерпретация иерархии в retrieval setting |
| 6.4 | Subgroup analysis | Кому помогает больше |
| 6.5 | Семантика уровней | Что кодирует иерархия |

### Последовательность выполнения

**Phase 1 — Foundation (blocker для всего):** 1.1 → 1.2 (параллельно) → 1.3

**Phase 2 — Codes:** 2.1 → 2.2 (fix D, K)

**Phase 3 — Sanity:** 3.1, 3.2, 3.3 (параллельно, все дешёвые)

**Phase 4 — Core improvement tests:** 4.1 → 4.2, 4.3 (параллельно) — главная часть работы

**Phase 5 — Secondary core:** 5.1 → 5.2

**Phase 6 — Extensions (по приоритету):** 6.1 → 6.2 → 6.3 → 6.4 → 6.5

### Критерий «работа удалась»

**Минимальный положительный результат:**

1. **RQ1 подтверждён** (§4.1): semantic codes сохраняют ≥ 90% качества SASRec-Content при прямой замене, и значимо лучше replace с random codes.
2. **RQ2 подтверждён** (§4.3): Fusion-B значимо лучше SASRec-ID baseline **и** значимо лучше Fusion-B с random codes.
3. **RQ3 хотя бы обсуждён** (§5.1): hierarchical показан как минимум не хуже flat при равном бюджете.

**Интерпретация разных исходов:**

- (1) + (2): semantic user IDs — компактная замена continuous представления, **и** дополняют collaborative SASRec. Сильный положительный результат.
- (1) без (2): коды работают как **substitute** (сохраняют качество), но не как **enhancement** (не добавляют сигнала поверх). Всё ещё valid contribution — важно для compact user representation.
- Ни (1), ни (2): semantic user IDs не показывают practical benefit в sequential recommendation на Yambda. Это valid negative result, который очерчивает границы применимости идеи и указывает направления для будущей работы.
