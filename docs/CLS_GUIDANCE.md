# CLS mean-diff guidance во внутренних активациях SHIFT

Код для нового эксперимента: **без обратимого адаптера и без обученного
предсказателя CLS**. Используется настоящий DINO CLS. Локально проверены только
CPU-тесты, включая маленькие FLUX/DINO со случайными весами. Предобученные модели,
датасет и реальные генерации здесь не запускались.

## Что сравниваем

Во всех режимах используется один и тот же CLS mean diff по train-парам и одна
функция потерь от one-step оценки. Различается только оптимизируемая переменная:

| Конфиг | Где изменяются тензоры |
| --- | --- |
| `configs/cls_guidance.json` | Все image tokens после одного double-блока (по умолчанию блок 9) |
| `configs/cls_velocity_guidance.json` | Предсказанный velocity на выходе трансформера |
| `configs/cls_guidance_all_steps.json` | Image tokens блока 9, каждый шаг, без потолка RMS, штрафа и отбора итерации |
| `configs/cls_velocity_guidance_all_steps.json` | Velocity, каждый шаг, без потолка RMS, штрафа и отбора итерации |
| `configs/cls_joint_all_blocks_all_steps.json` | Совместные поправки ко всем double-блокам на каждом шаге, без потолка RMS, штрафа и отбора итерации |

В первых двух конфигах шаг 0, 20 Adam-обновлений, alpha = 0, 0.5, 1.
Блок 9 — стартовый выбор для проверки, а не установленный оптимальный блок.
`blocks: "all"` при `block_mode: "independent"` (по умолчанию) означает отдельные
прогоны для каждого блока. При `block_mode: "joint"` все выбранные блоки изменяются
в одной генерации. В velocity-режиме блок не выбирается.
Веса FLUX, VAE, DINO заморожены. ROI глаз не используется: CLS — глобальная цель,
поправка разрешена на всех image-токенах.

## 0. Подготовка сервера

Все команды ниже выполняются из корня репозитория в Python-окружении сервера.
Для нового клона:

```bash
git clone --branch feat/dino-invertible-adapter https://github.com/ControlGenAI/SHIFT.git SHIFT-cls
cd SHIFT-cls
```

Если SHIFT уже склонирован, перейдите в его каталог и обновите ветку:

```bash
git fetch origin
git switch feat/dino-invertible-adapter
git pull --ff-only origin feat/dino-invertible-adapter
```

Сначала установите CUDA-версию PyTorch и torchvision под сервер, затем зависимости
SHIFT. Если используется готовое окружение SHIFT, активируйте его:

```bash
python -m pip install -r requirements.txt
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('BF16:', torch.cuda.is_bf16_supported())"
```

Для оптимизации оба значения должны быть `True`: FLUX и VAE загружаются в BF16.
На первом запуске потребуются указанные в конфиге веса FLUX и DINO из Hugging Face
или их локальный кеш; доступ к репозиторию модели должен быть настроен на сервере.
Сбор датасета и обучение адаптеров команды ниже не запускают.

`out/dataset200` — путь к **уже подготовленному** датасету на сервере, его нет в Git.
Замените этот путь на свой во всех командах. Нужны `dataset.json` и файлы, на которые
он ссылается: `features.pt` с CLS для `--cached-cls` либо PNG для пересчёта CLS.
Для сравнения в `dataset.json` должны быть положительные test-примеры с prompt/seed;
их пары и seed должны отличаться от train. При отсутствии test используйте val
для настройки, а окончательное сравнение проводите на отдельном test-наборе.

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
нельзя. В differentiable path нет PIL или detach изображения. Режим
`decode_mode: "pipeline"` использует такое же приведение dtype и ограничение RGB
в [0,1], как выход SHIFT, сохраняя граф до поправок.

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

После первого прогона проверьте в `out/cls_guidance_activation/` и
`out/cls_guidance_velocity/`:

- `*_baseline.png` и вариант с `alpha=0`: в JSON его `baseline_pixel_max_abs`
  должен быть 0 при воспроизводимом запуске с тем же seed.
- JSON ненулевого alpha: `selected_iteration`, `baseline_semantic_loss`,
  `final_semantic_loss`, `selected_relative_rms`. Если `selected_iteration=0`
  и RMS равен 0, улучшение не было выбрано и сохранился baseline.
- Финальные PNG: удаление очков, лицо и остальные детали. Снижение one-step loss
  само по себе не подтверждает нужного визуального изменения.

Для перебора всех double-блоков поставьте `"blocks": "all"` в копии
`configs/cls_guidance.json` и передайте её через `--config` перед `optimize`.
`"steps": [0]` оставьте для вмешательства только на первом шаге. Каждый блок
будет проверен в отдельной генерации, поэтому число прогонов существенно вырастет.

## Каждый шаг без ограничений на величину поправки

Для следующего эксперимента после `figures/cls_guidance/` подготовлены конфиги
`*_all_steps.json`. Прогоны блока 9 и velocity используют тот же mean diff и те же prompt/seed:

```bash
python -m src.dino_adapter.cls_experiment \
  --config configs/cls_guidance_all_steps.json optimize \
  --dataset out/dataset200 --split test --num-samples 1 \
  --direction out/cls_features/cls_direction.pt \
  --output out/cls_activation_all_steps

python -m src.dino_adapter.cls_experiment \
  --config configs/cls_velocity_guidance_all_steps.json optimize \
  --dataset out/dataset200 --split test --num-samples 1 \
  --direction out/cls_features/cls_direction.pt \
  --output out/cls_velocity_all_steps
```

| Параметр | Старые конфиги | `*_all_steps.json` |
| --- | --- | --- |
| `steps` | `[0]` | `[0,1,2,3]` |
| `iterations` | 20 | 20 на каждом шаге, всего 80 обновлений |
| `learning_rate` | 0.01 | 0.01 |
| `max_relative_rms` | 0.05 | `null`: нет проекции и проверки RMS-бюджета |
| `preservation_weight` | 1.0 | 0.0: в loss только CLS-цель |
| `selection` | `best` по умолчанию | `last`: последнее состояние после всех обновлений Adam |
| `save_step_predictions` | false по умолчанию | true |

В режиме `last` ухудшение loss не откатывает результат к baseline. В конфиге можно
независимо вернуть `selection: "best"`, положительный штраф или численный RMS-cap.
Если одновременно включены `last` и cap, превышение бюджета после BF16-округления
завершает прогон с ошибкой; скрытого выбора другой итерации нет.

Цель заново строится на каждом шаге из **текущей, уже изменённой траектории**:
`target = normalize(source_cls_at_this_step - alpha * direction)`. Она фиксирована
внутри 20 обновлений данного шага. Alpha применяется на каждом шаге, поэтому это
не один сдвиг, распределённый между четырьмя sigma. Adam начинает с новых моментов
на каждой sigma. Alpha=0 по-прежнему полностью обходит внутреннюю оптимизацию.

Для ненулевых alpha сохраняются `*_step0_before.png`, `*_step0_after.png` и такие же
пары для шагов 1–3. Это one-step оценки до/после вмешательства **на данном шаге**;
`before` следующих шагов уже содержит результат предыдущих вмешательств. PNG
проходят обычный postprocess с обрезкой в диапазон изображения, который не входит
в loss. Эти файлы нужны, чтобы проверить, появляются ли изменения очков в самих
оценках и сохраняются ли они в финальном PNG.

В каждой итерации JSON записаны `gradient_rms`, `gradient_nonzero` (если далее есть
Adam-обновление), `fp32_relative_rms`, `actual_relative_rms`, `velocity_relative_rms`
и `projected`. В итоговой записи каждой sigma сохранены реальные параметры cap,
штрафа, выбора состояния, LR, числа обновлений и dtype. В конфигурации без потолка
`projected` всегда false; градиенты не клипуются, исходная норма H/v не восстанавливается.

### Сравнение с `dino_guidance`

Проверена локальная версия `dino_guidance` на `c34644d`. В старых pilot-скриптах
`bounded_correction` масштабирует градиент относительно RMS шага Euler
`(sigma_next - sigma) * velocity`; backtracking/best-of-nine может отклонить поправку.
В активном `optimization.py::adam_refine` patch-only режима этих операций нет:
обновляется FP32 master-латент перед FLUX, берётся последнее состояние Adam.
В `configs/adam_local_patch.json` это отражено как `update_rms_ratio: null` и
`candidate_selection: "none; every Adam step is taken"`.

Новый SHIFT-режим отключает аналогичные ограничения величины и выбора результата,
но оптимизируемая переменная другая: post-block H или выходной v. Параметризация
остаётся `tensor = original + RMS(original) * u`, так что LR относится к u;
численно равный LR из `dino_guidance` не означает равного изменения тензора.
FP32 u сохраняется между inner-итерациями, BF16 используется при подстановке в модель.
Малые изменения могут округляться до нуля в отдельных forward, но накапливаются в u.
Проверки NaN/Inf сохранены и завершают некорректный прогон с ошибкой.

Множитель `(sigma_next - sigma)` при применении velocity — часть шага scheduler,
он не удаляется и повторно внутри оптимизатора не применяется. Оптимизация латента
до FLUX меняет также исходную точку этого шага; поэтому прошлый латентный guidance
и нынешний velocity-guidance не эквивалентны даже без clipping.

В этом эксперименте меняются и расписание, и ограничения. Чтобы отделить эффект
расписания, сделайте копию старого конфига и поменяйте только `steps` на `[0,1,2,3]`.
Снятие потолка допускает большие изменения изображения; это проверка влияния
ограничений, а не подтверждённый режим сохранения лица.

## Совместная оптимизация всех double-блоков

`configs/cls_joint_all_blocks_all_steps.json` включает `blocks: "all"`,
`cls_optimization.block_mode: "joint"` и `space: "activation"`. Список double-блоков
берётся из загруженного FLUX. Можно задать подмножество, например `blocks: [3, 9, 15]`.
Это одно совместное условие на каждый alpha; число генераций не умножается на число блоков.

```bash
python -m src.dino_adapter.cls_experiment \
  --config configs/cls_joint_all_blocks_all_steps.json optimize \
  --dataset out/dataset200 --split test --num-samples 1 \
  --direction out/cls_features/cls_direction.pt \
  --output out/cls_joint_all_blocks_all_steps
```

На каждой sigma сначала делается forward без новых поправок, чтобы получить CLS-цель
и масштабы `s_i = RMS(H_i_baseline)` каждого блока. Затем создаётся отдельный FP32
тензор `u_i[B,N,C]` для каждого выбранного блока. В каждом inner forward:

```text
text_i, image_i = double_block_i(text_previous, image_previous)
image_i_edited = image_i + s_i * u_i
                          ↓ следующие double- и single-блоки
velocity → x0_hat → VAE decode → DINO CLS → один общий loss
```

Поправка добавляется к **текущему** выходу блока. Поэтому изменение раннего блока
влияет на поздние, и общий loss дифференцируется по всем `u_i` через связанную цепочку.
Подстановка сохранённых независимых `H_i_baseline + s_i*u_i` во все блоки разорвала бы
этот путь по image-ветке. Пространственные image-токены не усредняются: у каждого
своя поправка. CLS используется только для общей цели в конце вычисления.
Text-выход каждого hook передаётся дальше как получен; в последующих блоках он
естественно зависит от изменённых image-токенов через attention.

Один Adam одновременно обновляет все `u_i`: 20 обновлений на каждой из четырёх sigma,
всего 80 совместных обновлений. На следующей sigma создаются новые нулевые `u_i`
и новые моменты Adam; предыдущие изменения уже присутствуют в текущем латенте.
Масштаб `s_i` фиксирован внутри шага и задаёт единицы LR, не ограничивая амплитуду.
Конфиг использует `max_relative_rms: null`, `preservation_weight: 0`, `selection: "last"`.
Веса модели не обучаются. Velocity получается из изменённого forward; отдельной
оптимизируемой поправки на выходной velocity в этом режиме нет.

Hook-и сохраняются на время backward, включая повторные вычисления checkpointing,
и снимаются до обновления параметров Adam. Поддержан стандартный non-reentrant
checkpointing Diffusers. Локальные тесты сверяют **градиенты каждого блока** с прямыми
добавлениями внутри FLUX без hooks/checkpointing, в FP32 и BF16. Они также проверяют
совпадение с обычным Adam без ограничений, точный alpha=0 и один scheduler step на sigma.

В JSON `blocks` содержит фактические индексы, а `block_mode` равен `joint`.
Каждая итерация содержит `per_block`: градиент, FP32 RMS поправки и фактический RMS
добавления после округления в dtype модели. Итог шага содержит `selected_per_block`.
RMS считается относительно исходного масштаба данного блока и описывает **местное
добавление**, не включая пришедшее из ранних блоков изменение. Общие RMS-поля —
корень из среднего квадратов RMS всех выбранных блоков. `velocity_relative_rms`
показывает суммарное влияние на выход FLUX. Если вручную включить cap, он применяется
к каждому блоку отдельно; включённый штраф усредняется по блокам.

Оптимизация ранних блоков требует backward почти через весь FLUX, а FP32 поправки,
их градиенты и моменты Adam хранятся для всех выбранных блоков. Поэтому потребление
памяти выше, чем при вмешательстве в один поздний блок. На настоящем FLUX этот режим
здесь не запускался; улучшение удаления очков и сохранения лица ещё нужно проверить
по финальным изображениям при сопоставимой силе редактирования.

## Как считается градиент

Для режимов одного блока и velocity на выбранном шаге фиксируем исходный латент `z_sigma`, conditioning и исходный
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

Если `max_relative_rms` задан числом, поправка проецируется в RMS-бюджет с проверкой
фактического изменения после BF16-cast; `null` отключает это ограничение.
При `selection: "best"` из состояний 0..N выбирается лучшее **по полной функции
потерь**. Исходный baseline тоже кандидат: если улучшения нет, возвращается исходный
velocity. При `selection: "last"` возвращается состояние N. Ни один из режимов не
выбирает лучшую финальную картинку и не гарантирует сохранение личности.

Итоговый velocity — именно тот, который был вычислен и оценён для выбранного H.
Scheduler выполняет **один** шаг после inner loop; оптимизация не меняет latent,
conditioning, параметры моделей или scheduler state. Alpha=0 обходит оптимизацию
и должен воспроизводить обычный baseline с тем же seed. Используется no_grad на
внешнем pipeline и enable_grad внутри; inference_mode для этого пути запрещён.

## Параметры и диагностика

В `cls_optimization`:

- `steps`: индексы шагов (по умолчанию `[0]`).
- `iterations`: число Adam-обновлений (20), после последнего есть оценка результата.
- `learning_rate`: шаг Adam в выбранных единицах поправки (0.01).
- `correction_scaling`: `rms` (по умолчанию, `H + RMS(H0)*u`) или `none` (`H + delta`).
- `match_rms_adam`: контроль пересчёта LR и epsilon при прямой поправке; по умолчанию false.
- `decode_mode`: `pipeline` по умолчанию; `legacy_fp32_unclipped` воспроизводит
  прежний FP32 decode и RGB до clipping, использованные в экспериментах до исправления.
- `first_update_probe`: дополнительные знаковые множители первого направления
  Adam для диагностики. Пустой список по умолчанию; непустой должен включать 0.
  При нуле повторяется backward и измеряется воспроизводимость градиентов.
- `preservation_weight`: вес штрафа на изменение выбранного тензора; 0 отключает штраф.
- `max_relative_rms`: максимальный RMS изменения относительно исходного тензора; null отключает потолок.
- `selection`: best (по умолчанию, с baseline-кандидатом) или last (последнее состояние Adam).
- `save_step_predictions`: сохранять one-step PNG до/после каждого управляемого шага (false по умолчанию).
- `gradient_checkpointing`: checkpointing FLUX (true).
- `space`: activation или velocity.
- `block_mode`: independent (по умолчанию, отдельная генерация для каждого блока)
  или joint (совместная оптимизация выбранных double-блоков; только activation).

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
Кеширование forward трансформера нужно отключить: во внутреннем цикле активации
меняются при той же sigma, поэтому повторное использование кеша недопустимо.
Индексы шагов должны быть целыми и входить в фактическое расписание генерации;
`resolution=(height, width)` у guidance должен совпадать с размером pipeline.

RMS здесь считается по всему тензору изображения, а не отдельно по каждому токену.
Он не запрещает сосредоточить изменение в небольшой области лица. Кроме того,
DINO в режиме `pipeline` получает RGB после штатного clipping, до округления в PNG.
На последнем шаге с конечной sigma=0 эта RGB-картинка должна совпадать с выходом
пайплайна; этот контракт проверяется отдельно в FP32/BF16. На ранних шагах это
по-прежнему one-step оценка, и последующие шаги могут изменить её свойства.
`legacy_fp32_unclipped` оставлен для воспроизведения прежних экспериментов.
Проверка финальной картинки и сохранения лица остаётся обязательной частью
эксперимента; для этого одной CLS-проекции недостаточно.

CPU-проверки без pretrained весов:

```bash
python -m pip install pytest
python -m pytest tests -q
python -m src.dino_adapter.cls_experiment --help
```

`tests/test_cls_pipeline.py` проверяет именно цикл SHIFT с маленькими случайными
FLUX/VAE/DINO, включая прямоугольное изображение, FP32/BF16, checkpointing,
нулевое вмешательство и число шагов scheduler. Текст подаётся готовыми embeddings;
тесты не скачивают веса. Форматы PNG/JSON/PT проверяются во временном каталоге
на выходах маленьких случайных моделей; реальные данные не собираются.
`tests/test_cls_joint_guidance.py` дополнительно проверяет совместные вмешательства,
градиенты всех блоков при checkpointing и соответствие обычному Adam.

Сравнение RMS-масштабирования с прямыми поправками: три конфига, команда для
кластера и интерпретация результатов описаны в
[CLS_SCALING_COMPARISON.md](CLS_SCALING_COMPARISON.md).

Разбор результатов `1a0f4f2`, исправление несовпадения decode и короткая проверка
градиента на кластере: [CLS_SCALING_AUDIT.md](CLS_SCALING_AUDIT.md).

Направление отдельно на каждой sigma через `image -> VAE -> noise -> one-step -> DINO CLS`,
сравнение prompts и проверки BF16/FP32: [CLS_NOISED_MEAN.md](CLS_NOISED_MEAN.md).

Разбор результатов `22c0ae9`, сдвиг только вдоль концепта без штрафа на остальные
координаты CLS, проверка более сильных alpha и `audit-run` без GPU:
[CLS_REMOVAL.md](CLS_REMOVAL.md).
