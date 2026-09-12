# CLS mean-diff через one-step оценку на разных уровнях шума

Результаты этих конфигов из `22c0ae9` и следующий тест на удаление очков:
[CLS_REMOVAL.md](CLS_REMOVAL.md). Описанный ниже нормированный target остаётся
режимом по умолчанию; новый `cls_loss=projection` задаёт цель только по проекции.

Этот эксперимент сравнивает прежнее направление из чистых изображений с
направлением из представления, на котором работает one-step objective.
Извлечение требует FLUX, VAE и DINO на GPU; локально используются только тесты
на маленьких случайных CPU-моделях. Обучения нового предиктора CLS здесь нет.

## Как считается направление

Для каждой картинки пары и каждого выбранного шага:

```text
image -> VAE encoder -> z0
z_sigma = (1 - sigma) * z0 + sigma * epsilon
velocity = FLUX(z_sigma, timestep, prompt)
x0_hat = z_sigma - sigma * velocity
x0_hat -> VAE decoder -> штатный RGB [0,1] -> DINO CLS
```

Это flow-matching noise, а не DDPM-формула и не прибавление шума к RGB.
VAE использует `latent_dist.mode()`, затем `(latent - shift_factor) * scaling_factor`
и штатную упаковку латентов. Это исключает дополнительную случайность posterior.
`scale_noise` вызывается на отдельном свежем scheduler: после `set_begin_index(0)`
его реализация могла бы брать sigma первого шага вместо запрошенной.
Формула и поведение сверены с
[diffusers v0.38.0](https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py#L175).

У обеих картинок пары одинаковый `epsilon` для данного повтора. Этот же шум
используется на всех уровнях; у разных пар и повторов разные детерминированные seeds.
Расписание берётся из той же функции, что и генерация SHIFT, включая shift scheduler.
Для текущего четырёхшагового Schnell это sigma `[1, 0.75, 0.5, 0.25]`.

Для каждого уровня сохраняется отдельное направление:

```text
d_sigma = mean_train_pairs,repeats(CLS_with_concept - CLS_without_concept)
```

Каждый CLS нормализован в FP32. Сначала считаются наблюдения по повторам, затем
средние; усреднённые векторы повторно не нормализуются. Сам mean-diff также не
приводится к единичной длине. Val/test не участвуют в расчёте направления.

При оптимизации выбирается `d_sigma` именно для текущего шага, sigma и timestep.
Цель остаётся `normalize(CLS_source - alpha * d_sigma)`. Источник пересчитывается
из текущего латента перед оптимизацией каждого шага. Таблица от другого расписания,
разрешения, модели, dtype или decode отклоняется вместо молчаливой подстановки.

## Два режима conditioning

| Конфиг | Prompt при извлечении | Что происходит при sigma=1 |
| --- | --- | --- |
| `configs/cls_joint_noised_mean.json` | Собственный prompt каждой картинки (`paired_prompts`) | Входные латенты пары совпадают; разница CLS определяется разными prompts |
| `configs/cls_joint_noised_mean_shared_prompt.json` | Общий `A front-facing portrait photograph of a person.` | При одинаковом шуме и детерминированном forward картинки совпадают, mean-diff равен нулю |

При sigma=1 исходная картинка полностью теряется: `z_sigma = epsilon`. Поэтому
общий prompt на этом уровне не может дать направление, вызванное различием картинок.
Точный нулевой mean-diff даёт явный bypass с `reason: "zero_mean_diff"` в логе.
Ненулевые направления не отбрасываются по порогу и не заменяются чистым направлением.
В `paired_prompts` эффект текста присутствует и на остальных уровнях.
`guidance_scale=0` у Schnell не отключает текстовое conditioning.

Оба конфига меняют все image-токены всех double-блоков на всех четырёх шагах:
20 Adam-обновлений на шаг, LR=0.01, `correction_scaling=none`,
`preservation_weight=0`, `max_relative_rms=null`, `selection=last`.
Настройки сохранены для сравнения с прошлым raw-прогоном; LR=0.01 ещё не признан удачным.
Таблица направлений также работает с одиночным блоком и `space=velocity`.
Она предназначена для `objective=one_step`; финальный objective использует CLS финальной картинки.

## Запуск на кластере

После переноса изменений в приватное рабочее окружение, из корня SHIFT с активным venv:

```bash
python -m src.dino_adapter.cls_experiment \
  --config configs/cls_joint_noised_mean.json \
  extract-noised --dataset out/dataset200 --output out/cls_noised_paired

python -m src.dino_adapter.cls_experiment audit \
  --features out/cls_noised_paired/cls_features.pt \
  --direction out/cls_noised_paired/cls_direction.pt \
  --output out/cls_noised_paired_audit.json

python -m src.dino_adapter.cls_experiment \
  --config configs/cls_joint_noised_mean.json \
  optimize --dataset out/dataset200 --split test --num-samples 1 \
  --direction out/cls_noised_paired/cls_direction.pt \
  --output out/test_cls_noised_paired
```

Для общего prompt повтори команды с `cls_joint_noised_mean_shared_prompt.json`
и новыми каталогами, например `out/cls_noised_shared` и `out/test_cls_noised_shared`.
Для контроля с чистым направлением используй тот же конфиг оптимизации и прежний
`out/cls_features/cls_direction.pt`, сохраняя результат в отдельный каталог.
Не сравнивай старый `legacy_fp32_unclipped` прогон с новым без учёта смены decode.

По умолчанию 4 повтора шума. Для 200 пар это 400 VAE-encode и
`400 * 4 * 4 = 6400` one-step FLUX/VAE/DINO вычислений, плюс CLS исходных картинок.
Для первого smoke-прогона можно явно поставить `noise_repeats: 1` в копии конфига;
это изменит оценку направления, поэтому укажи новый каталог.

На каждую картинку сохраняется кеш в `samples/`. Прерванное извлечение можно
продолжить той же командой с `--resume`; конфиг, изображения, код и окружение должны
совпадать. Средние из завершённого кеша пересчитываются командой `mean` без GPU.
`previews/` содержит one-step картинки первых двух пар, по одному повтору на sigma.
Проверь, сохранился ли концепт в этих оценках. `summary.json` содержит нормы направлений;
`audit` проверяет соответствие кешу и разделимость на train/val/test отдельно по sigma.

## Проверка dtype и ограничений

`model_dtype` и `vae_dtype`: `bfloat16` или `float32`. При отсутствии полей остаётся
прежний BF16. При отдельном FP32 VAE веса сразу загружаются в FP32; upcast уже
округлённых при загрузке BF16-весов не используется. Перед VAE scale/shift латенты
приводятся к dtype VAE и в objective, и в финальном decode SHIFT.

DINO, CLS, target, loss, поправки и Adam state используют FP32. Внешний autocast
отключается внутри DINO и при расчёте градиентов поправок; приведение уже рассчитанного BF16 CLS к FP32 не восстановило
бы исходную точность. Forward FLUX и внесение поправки в image-токены используют
dtype FLUX. Поэтому FP32-поправка всё ещё может округлиться при внесении в BF16:
например, `BF16(32 + 0.01) == 32`, хотя autograd пропускает градиент через cast.
Это одна из причин, по которой очень маленькая проба может дать плато loss.

Для короткой проверки первого обновления с прежним чистым направлением:

```bash
python -m src.dino_adapter.cls_experiment \
  --config configs/cls_joint_debug_vae_fp32.json \
  optimize --dataset out/dataset200 --split test --num-samples 1 \
  --direction out/cls_features/cls_direction.pt --output out/cls_debug_vae_fp32
```

Сравни с `cls_joint_debug_first_update.json` (BF16) и, если хватает памяти,
`cls_joint_debug_fp32.json` (FLUX, text encoders и VAE в FP32). Полный FP32 FLUX
требует больше памяти. Для оптимизации с шумным направлением при смене dtype
нужно также заново извлечь это направление в том же dtype.

В старых `cls_guidance.json` и `cls_velocity_guidance.json` остаются явные
ограничения: cap=0.05, preservation=1, selection=best, только шаг 0.
Без `--config` CLI выбирает первый из них. При старте теперь печатаются применённые
настройки. В новых конфигах нет проекции поправок, штрафа, gradient clipping,
автоматического уменьшения LR или отката к baseline. RGB clipping остаётся частью
штатного декодирования; нормировка CLS и target остаётся частью выбранного loss.

Совпадение one-step representation не гарантирует редактирование без артефактов:
зашумлённые реальные латенты всё ещё могут отличаться по распределению от состояний
генерации, а глобальный CLS-loss не задаёт отдельного требования сохранить личность.
Оцени финальные картинки и независимые метрики, а не только снижение loss.

## Локальная проверка

152 CPU-теста прошли без загрузки pretrained-весов. Новые проверки сравнивают
one-step CLS с настоящим Euler-переходом в sigma=0, включая динамический shift,
нецелые timesteps, BF16 и отдельный FP32 VAE; проверяют шум пары, train-only средние,
выбор направления по sigma, кеш/возобновление, audit и последующую оптимизацию.
Отдельный тест воспроизвёл изменение CLS/градиентов под внешним autocast и проходит
после отключения autocast в DINO forward и расчёте градиентов оптимизатора.
Старые проверки градиентов всех 19 double-блоков и отсутствия ограничений в режимах
`last`/`null` также проходят. Результаты этого нового эксперимента на GPU ещё не получены.
