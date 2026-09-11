# CLS mean-diff guidance во внутренних активациях SHIFT

Код для нового эксперимента: **без обратимого адаптера и без обученного
предсказателя CLS**. Используется настоящий DINO CLS. Локально проверены только
CPU-тесты, включая маленькие FLUX/DINO со случайными весами. Предобученные модели,
датасет и реальные генерации здесь не запускались.

## Что сравниваем

В обоих режимах используется один и тот же CLS mean diff по train-парам и одна
функция потерь от one-step оценки. Различается только оптимизируемая переменная:

| Конфиг | Где изменяются тензоры |
| --- | --- |
| `configs/cls_guidance.json` | Все image tokens после одного double-блока (по умолчанию блок 9) |
| `configs/cls_velocity_guidance.json` | Предсказанный velocity на выходе трансформера |

Шаг по умолчанию 0, 20 Adam-обновлений, alpha = 0, 0.5, 1.
Блок 9 — стартовый выбор для проверки, а не установленный оптимальный блок.
`blocks: "all"` в activation-конфиге означает отдельные прогоны для каждого блока.
В velocity-режиме блок не выбирается. Одновременных вмешательств во все блоки нет.
Веса FLUX, VAE, DINO заморожены. ROI глаз не используется: CLS — глобальная цель,
поправка разрешена на всех image-токенах.

## 1. Извлечение CLS и вычисление mean diff

Если есть `out/dataset200` из эксперимента адаптеров, в его `features.pt` уже есть
настоящие CLS DINO. Следующая команда **не загружает модели и не использует GPU**:

```bash
python -m src.dino_adapter.cls_experiment --device cpu extract \
  --dataset out/dataset200 --cached-cls \
  --output out/cls_features
```

Чтобы пересчитать CLS по картинкам с DINO из конфига, уберите `--cached-cls`:

```bash
python -m src.dino_adapter.cls_experiment extract \
  --dataset out/dataset200 --output out/cls_features_recomputed
```

Для своих картинок поддерживается JSONL с путями относительно manifest:

```json
{"pair_id":"p01","label":1,"split":"train","image":"images/p01_with.png","seed":101}
{"pair_id":"p01","label":0,"split":"train","image":"images/p01_without.png","seed":101}
```

```bash
python -m src.dino_adapter.cls_experiment extract \
  --manifest data/cls_pairs.jsonl --output out/cls_features_custom
```

Каждый CLS нормализуется по L2. Направление — средняя разность
`CLS_with - CLS_without` по **полным train-парам**. Сам mean diff не нормализуется
до единичной длины. Validation/test не участвуют. Seed проверяется на пересечение
split, когда указан; реальные идентичности и качество пар всё равно нужно проверить.

Сохраняются `cls_features.pt`, `cls_direction.pt`, `summary.json` с числом пар и
нормой направления. В одном `extract` выполняются и извлечение, и mean diff.
Пересчитать mean diff отдельно из сохранённых фич можно без моделей:

```bash
python -m src.dino_adapter.cls_experiment mean \
  --features out/cls_features/cls_features.pt \
  --output out/cls_direction_recomputed.pt
```

В API `DinoFeatures.cls_from_rgb(rgb)` принимает BCHW RGB в номинальном [0,1]
и возвращает нормализованный CLS **с градиентом**. `tensor_features` возвращает
patches и CLS за один forward. CLS берётся из `last_hidden_state[:,0]`, а не из
среднего patch tokens. Из одних сохранённых patch tokens настоящий CLS восстановить
нельзя. В differentiable path нет PIL, clamp или detach изображения.

## 2. Запуск оптимизации на GPU-сервере

Внутренние активации, один положительный test-промпт:

```bash
python -m src.dino_adapter.cls_experiment optimize \
  --dataset out/dataset200 --split test --num-samples 1 \
  --direction out/cls_features/cls_direction.pt \
  --output out/cls_guidance_activation
```

Тот же эксперимент на выходном velocity:

```bash
python -m src.dino_adapter.cls_experiment \
  --config configs/cls_velocity_guidance.json optimize \
  --dataset out/dataset200 --split test --num-samples 1 \
  --direction out/cls_features/cls_direction.pt \
  --output out/cls_guidance_velocity
```

`--config` и `--device` ставятся **до** подкоманды. Для своих prompt/seed можно
заменить `--dataset` на `--prompts data/cls_test_prompts.jsonl` с записями:

```json
{"id":"test01","pair_id":"heldout01","prompt":"A portrait of a person wearing eyeglasses","seed":510001}
```

Выборка по умолчанию ограничена первым примером. Для расширения укажите
`--num-samples N`. Для повторов используйте новый output: файлы не перезаписываются.

## Как считается градиент

На выбранном шаге фиксируем исходный латент `z_sigma`, conditioning и исходный
post-block тензор `H0` (или velocity `v0`). Baseline CLS:

```text
v0 = FLUX(z_sigma)
x0_hat = z_sigma - sigma * v0
c0 = normalize(DINO_CLS(VAE_decode(x0_hat)))
target = normalize(c0 - alpha * mean_diff)
```

Target detached и фиксирован в течение inner loop. При нескольких управляемых
шагах он заново строится на каждом шаге из текущего состояния траектории.
Положительный alpha удаляет очки, отрицательный добавляет.

Оптимизируем FP32 поправку `u`, а тензор для модели:
`H = H0 + RMS(H0)*u`. Forward hook заменяет только image-выход выбранного блока;
text-выход в точке вмешательства сохраняется. Последующие блоки могут менять обе
ветки естественным attention-взаимодействием. Они формируют velocity, затем
вычисляются `x0_hat`, VAE decode и DINO CLS. Градиент идёт к `u` через весь этот путь.

Префикс FLUX пересчитывается при каждом forward, но имеет замороженные
входы/веса; градиент начинается с подставленного H. Checkpointing последующих блоков
поддержан. В velocity-контроле оптимизируется `v=v0+RMS(v0)*u`; повторных FLUX
forward внутри оптимизации не требуется, градиент идёт через VAE и DINO.

```text
loss = 0.5 * ||CLS - target||²
     + preservation_weight * mean(((actual_model_tensor - original) / scale)²)
```

Поправка проецируется в RMS-бюджет. Из состояний 0..N выбирается лучшее **по полной
функции потерь**, с проверкой фактического RMS после BF16-cast. Исходный baseline
тоже кандидат: если улучшения нет, возвращается исходный velocity. Это не выбор
лучшей финальной картинки и не гарантия сохранения личности.

Итоговый velocity — именно тот, который был вычислен и оценён для выбранного H.
Scheduler выполняет **один** шаг после inner loop; оптимизация не меняет latent,
conditioning, параметры моделей или scheduler state. Alpha=0 обходит оптимизацию
и должен воспроизводить обычный baseline с тем же seed. Используется no_grad на
внешнем pipeline и enable_grad внутри; inference_mode для этого пути запрещён.

## Параметры и диагностика

В `cls_optimization`:
- `steps`: индексы шагов (по умолчанию `[0]`).
- `iterations`: число Adam-обновлений (20), после последнего есть оценка результата.
- `learning_rate`: шаг в нормализованной поправке u (0.01).
- `preservation_weight`: вес штрафа на изменение выбранного тензора (1.0).
- `max_relative_rms`: максимальный RMS изменения относительно исходного тензора (0.05).
- `gradient_checkpointing`: checkpointing FLUX (true).
- `space`: activation или velocity.

Одинаковый RMS-бюджет в H и velocity не означает одинакового эффекта на картинку.
Сравнивать сохранение свойств нужно при сопоставимом удалении очков.

Сохраняются baseline PNG, изображения каждого условия, JSON с loss всех итераций,
выбранной итерацией, фактической RMS и изменением velocity. `*_cls_targets.pt`
содержит source/target/selected CLS каждой управляемой sigma. В provenance записаны
hash направления и версии библиотек. `generations.json` содержит также DINO CLS
проекцию **финальной** картинки относительно baseline и отличия в пикселях.
Это позволяет заметить ситуацию, когда loss one-step оценки падает, но нужного
изменения в финальном изображении нет. CLS-проекция — не независимый детектор очков.

Первый smoke test: один prompt, один блок, alpha=0 и один ненулевой alpha.
Проверить alpha=0 vs baseline, ненулевой градиент/выбранное обновление, loss и
финальное изображение. В частности, ограничения RMS не доказывают реалистичность.
Направление от финальных картинок может плохо переноситься на раннюю x0-оценку.
True CFG пока не поддерживается. Полная цепочка FLUX/VAE/DINO с backward требует
существенной VRAM; работа на настоящих весах должна быть проверена на сервере.

CPU-проверки без pretrained весов:

```bash
python -m pytest tests -q
python -m src.dino_adapter.cls_experiment --help
```
