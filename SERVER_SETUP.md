# Запуск на GPU-сервере

Пошаговая инструкция для чистой установки на арендованном сервере с PyTorch + CUDA.
Все команды — от корня репозитория.

## 0. Пререквизиты

- Доступ по SSH к серверу (`ssh -p <PORT> root@<IP>`).
- Токен HuggingFace с доступом на чтение приватного датасета `<user>/yambda-semantic-user-ids`.
- API-ключ W&B (если планируется логирование — `wandb.mode: online`).
- Примонтированный Network Volume желательно (иначе при прерывании spot-инстанса всё теряется).

## 1. Клонирование репозитория

```bash
git clone <repo-url> exp
cd exp
```

## 2. Установка `uv` и зависимостей

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

uv sync
```

`uv` установит Python 3.12.9 (из `.python-version`) и все пакеты из `uv.lock`,
включая PyTorch с CUDA-колёсами.

## 3. Проверка GPU

```bash
uv run python -c "import torch; print('cuda:', torch.cuda.is_available()); \
print('device:', torch.cuda.get_device_name(0)); \
print('bf16:', torch.cuda.is_bf16_supported())"
```

Ожидается:
```
cuda: True
device: NVIDIA H100 80GB HBM3
bf16: True
```

## 4. Скачивание датасета с HF Hub

```bash
uv run hf auth login          # paste the read token
uv run python scripts/10_pull_from_hub.py
```

Скрипт восстанавливает:
- `data/processed/{train,val,test}.parquet`
- `data/processed/{train,val,test}_subsample_{10,1}pct.parquet`
- `data/processed/item_embeddings.parquet`
- `data/{splits_metadata,filter_stats,raw_stats}.json`

Если HF-username в репо отличается от того, под которым залогинен,
передай его явно:
```bash
uv run python scripts/10_pull_from_hub.py --username <hf-username>
```

## 5. Пересборка `item_id_map`

```bash
uv run python scripts/09_build_item_id_map.py
```

Создаёт `artifacts/item_id_map.json` из свежескачанного `train.parquet`
(~30 секунд).

## 6. Настройка W&B

```bash
uv run wandb login            # paste the API key
```

Если логирование не нужно — перед запуском передай `wandb.mode=offline`
или `wandb.enabled=false`.

## 7. Smoke-тест на 1%-подвыборке

Проверка, что GPU используется и пайплайн обучения работает целиком:

```bash
uv run python scripts/train_sasrec.py --config-name sasrec_id \
    data.split_set=subsample_1pct \
    trainer.epochs=2 trainer.seed=42 \
    wandb.mode=offline \
    run_name=smoke_gpu_id
```

Ожидается: несколько эпох проходят за минуту-две, в логе виден `device=cuda:0`,
в конце — финальные метрики.

## 8. Основные запуски — всё одним скриптом

`scripts/run_stage1.sh` последовательно прогоняет:
1. SASRec-ID × 3 сида (42, 43, 44)
2. SASRec-Content × 3 сида
3. `aggregate_sasrec.py` → `results/sasrec_summary.csv`

Каждый сид — отдельный Hydra job (не multirun), чтобы падение одного сида
не убивало очередь. Логи: master в `logs/stage1_<timestamp>.log`,
плюс per-run `logs/<config>_seed<N>_<timestamp>.log`.

Запуск в tmux, чтобы не потерять при отвале SSH:

```bash
tmux new -s train
scripts/run_stage1.sh
```

Выйти из tmux без убийства сессии: `Ctrl+b`, затем `d`. Вернуться: `tmux attach -t train`.

### Полезные переменные окружения

| Переменная | По умолчанию | Что делает |
|---|---|---|
| `SEEDS` | `42 43 44` | список сидов через пробел |
| `SPLIT` | `full` | `subsample_10pct` или `subsample_1pct` для dry-run |
| `SKIP_ID=1` | — | пропустить SASRec-ID (резюм после сбоя) |
| `SKIP_CONTENT=1` | — | пропустить SASRec-Content |
| `SKIP_AGGREGATE=1` | — | пропустить финальную агрегацию |
| `EXTRA` | `""` | любые Hydra overrides (напр. `trainer.batch_size=1024`) |

Примеры:
```bash
SPLIT=subsample_1pct SEEDS=42 scripts/run_stage1.sh          # smoke-test
SKIP_ID=1 scripts/run_stage1.sh                              # только content + aggregate
EXTRA="trainer.num_workers=16" scripts/run_stage1.sh         # больше DataLoader воркеров
```

Перед стартом скрипт проверяет, что CUDA доступен, и падает сразу, если нет.

### Ручной режим (если скрипт не подходит)

```bash
uv run python scripts/train_sasrec.py --config-name sasrec_id -m trainer.seed=42,43,44
uv run python scripts/train_sasrec.py --config-name sasrec_content -m trainer.seed=42,43,44
uv run python scripts/aggregate_sasrec.py
```

## 9. Агрегация результатов

`run_stage1.sh` вызывает её автоматически. Если нужно пересобрать после ручных
докачек или с другими сидами:

```bash
uv run python scripts/aggregate_sasrec.py --seeds 42 43 44
uv run python scripts/aggregate_sasrec.py --include-baselines   # добавить MostPop/Random
```

Результаты кладутся в `results/sasrec_id.csv`, `results/sasrec_content.csv`,
`results/sasrec_summary.csv`.

## 10. Дефолтные параметры уже подогнаны под H100

`configs/trainer/sasrec.yaml` на момент импорта настроен так:

| Параметр | Значение | Почему |
|---|---|---|
| `batch_size` | 512 | H100 80GB вытягивает; для MPS/CPU снизить до 256 |
| `eval_batch_size` | 512 | то же |
| `num_workers` | 8 | под 8–16 vCPU инстанса |
| `cudnn_deterministic` | false | +15% throughput; для mean±std по 3 сидам этого достаточно |

Если нужно переопределить под конкретный запуск — через CLI:
```bash
uv run python scripts/train_sasrec.py --config-name sasrec_id \
    -m trainer.seed=42,43,44 \
    trainer.batch_size=1024 trainer.num_workers=16
```

Если на H100 заметишь OOM с этими дефолтами — снизь `batch_size` до 384 или 256
(при длинной `max_seq_len=200` квадратичный attention + софтмакс-логиты
могут неожиданно съесть память).

## 11. Сохранение артефактов после обучения

После прогонов перенеси обратно на локальную машину (с локальной машины):

```bash
rsync -avz -e "ssh -p $PORT" \
    $REMOTE:/path/to/exp/saved/ \
    ~/Desktop/TermPaper/exp/saved/

rsync -avz -e "ssh -p $PORT" \
    $REMOTE:/path/to/exp/artifacts/user_vectors/ \
    ~/Desktop/TermPaper/exp/artifacts/user_vectors/

rsync -avz -e "ssh -p $PORT" \
    $REMOTE:/path/to/exp/results/ \
    ~/Desktop/TermPaper/exp/results/
```

## Про прерывание spot-инстанса

`SASRecTrainer` сохраняет `model_best.pth` после лучшей эпохи по `tuning_metric`.
Промежуточных per-epoch чекпоинтов нет — при прерывании в середине прогона
сид придётся начинать с нуля.

Если Network Volume подключён — положи туда `saved/` и `artifacts/`
(через симлинки или изменив пути). Тогда прерывание сохранит прогресс.
