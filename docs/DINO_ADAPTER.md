# Обратимый DINO-адаптер в SHIFT

Код подготовлен в ветке `feat/dino-invertible-adapter`. На локальной машине
не выполнялись генерации, сбор активаций или обучение на изображениях.

## Что реализовано

`src/dino_adapter/` подключается к **существующему `src.models.flux.FluxPipeline`**.
Модель SHIFT не заменяется. Используется forward hook на выходе целого
`transformer_blocks[b]`: `(txt_hidden, img_hidden)` → `(txt_hidden, img_new)`.
Attention-выходы и single-stream блоки не используются.

Конфиг по умолчанию: **все double-stream блоки, шаг 0**. Количество блоков читается
из модели. Для каждого блока обучается отдельный адаптер, и при тестировании
редактируется один блок за прогон. Это сравнение блоков, а не одновременное
вмешательство во все блоки. Вручную можно поставить `"blocks": [0, 1, 2]`.

- `adapter.py`: нелинейный RealNVP из чередующихся affine coupling layers.
- `hooks.py`: image-only post-block hook и счётчик timestep.
- `features.py`: полнокадровые DINO patch features, согласование сеток.
- `data.py`: формат сохранённых данных и **опциональный**, явно запускаемый сбор на сервере.
- `training.py`: обучение по готовым файлам, выбор checkpoint по validation loss.
- `directions.py`: направления из train-пар и контрольный постоянный вектор в h.
- `steering.py`: вмешательства, контрольные генерации и оценка сохранённых картинок.

Все image-токены сохраняются. Для обычного FLUX при 512×512 сетка h равна 32×32.
В DINOv2-large полный кадр уменьшается до 448×448, давая ту же сетку 32×32
при размере патча 14. Центрального crop нет. При различии сеток DINO-карта
интерполируется к сетке h, затем нормализуется по каналам.

Для каждого токена `F(h)=(z,r)`, где `dim(z)=DINO hidden_size`,
`dim(r)=FLUX hidden_size-dim(z)`. Нормализация h по обучающей выборке также
обратима и сохранена в checkpoint. Loss: среднее `(z-DINO_patch(x))²`.
Сетки не сжимаются; декодер — точный обратный проход тех же coupling layers.
`r` означает оставшиеся координаты, а не обученное разделение фона и концепта.

Стиринг: `h' = F⁻¹(z - alpha*v_glasses, r)`. Положительный alpha удаляет
очки, отрицательный добавляет. Направление — средняя парная разность
DINO(с очками) − DINO(без очков) на **train-парах**. Вне ROI направление нулевое.
ROI по умолчанию — прямоугольник глаз для фронтальных центрированных портретов;
это не детектор лица и не подходит автоматически для произвольной композиции.

## Обучение по уже сохранённым данным

Команды ниже предназначены для сервера. Из корня репозитория, в существующем
окружении SHIFT (`requirements.txt`; CUDA PyTorch устанавливается под сервер):

```bash
python -m src.dino_adapter train \
  --dataset experiments/dino_adapter/dataset \
  --output experiments/dino_adapter/adapters \
  --device cuda:0

python -m src.dino_adapter directions \
  --dataset experiments/dino_adapter/dataset \
  --output experiments/dino_adapter/directions.pt
```

`directions` — **legacy**: `z` из разности DINO-патчей изображения, `h` из post-block
активаций. Для стиринга через `F⁻¹` нужен вектор в пространстве адаптера:

```bash
python -m src.dino_adapter directions-adapter-z \
  --dataset experiments/dino_adapter/dataset \
  --adapters experiments/dino_adapter/adapters \
  --roi full --max-pairs 100 \
  --output experiments/dino_adapter/dirv3_full100.pt \
  --device cuda:0
```

Это `mean(F(h_with).z − F(h_without).z)` по train-парам плюс `z_mean` /
`h_mean` (один вектор, broadcast на все токены). Пример одновременного
вмешательства во все double-блоки на шаге 0 с norm-preserving edit:

`scripts/dino_adapter_demo_renorm.py` + конфиги
`configs/dino_adapter_renorm_demo.json` /
`configs/dino_adapter_global_mean_demo.json`. Классы `RenormAdapterEdit` /
`RenormImageEdit` и `MultiImageBlockHook` лежат в `src/dino_adapter/`.
Примерные полоски и `generations.json` — в `figures/dino_adapter_steering/`.
Обычный `steer` по-прежнему правит **один блок за прогон**.

`train` **не генерирует изображения и не загружает FLUX/DINO**: только адаптеры
и готовые CPU tensors. Обучение последовательное по блокам; на GPU в каждый
момент один адаптер и batch токенов. Статистика нормализации и направления
считаются только по train. Validation/test-пары и seed отделены от train.
Обучение не использует test. Checkpoint выбирается по validation alignment MSE.

Формат `dataset/dataset.json`:

```json
{
  "version": 1,
  "config": {"...": "полный configs/dino_adapter.json"},
  "grid": [32, 32],
  "blocks": [0, 1],
  "samples": [
    {
      "id": "pair_000_1", "pair_id": "pair_000", "label": 1,
      "split": "train", "seed": 42000, "prompt": "portrait with eyeglasses",
      "image": "pair_000_1/image.png",
      "features": "pair_000_1/features.pt",
      "blocks": {"0": "pair_000_1/block_0.pt", "1": "pair_000_1/block_1.pt"}
    }
  ]
}
```

Пример показывает структуру, не полный валидный датасет: нужны обе метки каждой
пары (`1` — с очками, `0` — без), train и val; для тестирования также test.
`block_N.pt` содержит tensor `[image_tokens, FLUX_channels]`, без усреднения.
`features.pt` содержит `{"patches": tensor[N,D], "cls": tensor[D]}` финального
изображения. Пути относительны каталогу dataset. Старые усреднённые SHIFT-векторы
не содержат необходимых данных для такого обучения.

## Подготовка данных на GPU-сервере, если их ещё нет

Этот шаг **не запускается автоматически**. Сначала подготовьте JSONL пар:
`configs/dino_adapter_pairs.example.jsonl` — небольшой шаблон (8 train, 2 val,
2 test), а не достаточный датасет для научного вывода. Расширьте его, сохраняя
разделение идентичностей и seed между split. Одинаковый seed в положительной
и отрицательной половинах пары уменьшает вариативность, но не гарантирует
совпадение личности или геометрии — пары нужно просмотреть.

```bash
python -m src.dino_adapter collect \
  --pairs configs/dino_adapter_pairs.example.jsonl \
  --output experiments/dino_adapter/dataset \
  --device cuda:0
```

За одну генерацию сохраняются выходы всех выбранных double-блоков на шаге 0.
После этапа FLUX его память освобождается, затем DINO обрабатывает финальные
изображения. Сохраняются полные активации, поэтому предусмотрите место на диске.
Незавершённый сбор имеет `collection_progress.json`, но не готовый `dataset.json`.

## Стиринг в SHIFT

```bash
python -m src.dino_adapter steer \
  --dataset experiments/dino_adapter/dataset \
  --adapters experiments/dino_adapter/adapters \
  --directions experiments/dino_adapter/directions.pt \
  --output experiments/dino_adapter/test_step0 \
  --device cuda:0
```

Для другого конфига ставьте `--config` **до** подкоманды:

```bash
python -m src.dino_adapter --config configs/dino_adapter.json steer --help
```

Для каждого положительного test-промпта: baseline без hook, затем каждый блок
отдельно и alpha из конфига. Для каждого alpha сохраняются adapter и constant
control. Стандартная сетка `[0, 0.5, 1, 2]`; нулевой alpha действительно выполняет
`F⁻¹(F(h))`, а не обходит адаптер. Сохраняются ошибки roundtrip до BF16-cast и
реальная RMS-величина изменения после cast обратно в dtype FLUX.

Constant control использует один фиксированный train-вектор
`mean(h_positive-h_negative)` для блока. Его величина подбирается под RMS
адаптерного вмешательства на том же изображении. Направление постоянно;
скаляр зависит от размера сравниваемого вмешательства. Это отдельный контроль
постоянного сдвига, без нормализации или классификаторного gating старого SHIFT.

В `generations.json` записаны block, alpha, seed, checkpoint hash и диагностика.
`visual_review.csv` предназначен для проверки очков, личности и посторонних изменений.
Промпт, шум, расписание и текстовая ветка одинаковы между условиями.
Hook снимается и при исключениях; недопустимые повторные вызовы (например true CFG)
явно отклоняются. Старые launchers и режимы SHIFT остаются доступны.

## Проверка последствий вмешательства

Отдельная команда читает только сохранённые изображения и загружает DINO:

```bash
python -m src.dino_adapter evaluate \
  --directions experiments/dino_adapter/directions.pt \
  --results experiments/dino_adapter/test_step0 \
  --device cuda:0
```

`metrics.json` содержит DINO-проекцию изменения вдоль направления очков,
сходство CLS с baseline и MSE вне ROI, а также отличие от baseline в пикселях.
Это **прокси-метрики**, а не независимый детектор очков или face-recognition score.
Они не доказывают сохранение личности; дополните их визуальной оценкой.

`matched_effect_comparisons.json` подбирает ближайший constant control по величине
изменения DINO-прокси атрибута и показывает остаточную разницу. При большой разнице
эффекты не сопоставимы: расширьте alpha sweep. Сравнение при одинаковой RMS само по
себе не означает одинаковую силу изменения атрибута.

Никакое превосходство адаптера пока не установлено. Проверяемая гипотеза:
нелинейное отображение координат даст более аккуратное изменение при сопоставимом
эффекте на атрибут. Alignment loss сам по себе этого не гарантирует.

## CPU-проверки без моделей

```bash
python -m pytest tests -q
python -m src.dino_adapter --help
```

Тесты используют маленькие искусственные tensors и fake-модули. Они не скачивают
веса, не собирают активации настоящего FLUX и не запускают генерации.

## Прямой стиринг image tokens без адаптера

`configs/image_tokens.json` включает самостоятельный контроль:
`h' = h - alpha * mean_train(h_with_glasses - h_without_glasses)`.
Берётся `img_hidden` после double-блока; все пространственные токены сохраняются,
направление вне той же ROI обнулено. Все double-блоки проверяются отдельно на
шаге 0. Текстовые токены не меняются. Обученные адаптеры не требуются.

```bash
python -m src.dino_adapter --config configs/image_tokens.json steer \
  --dataset experiments/dino_adapter/dataset \
  --directions experiments/dino_adapter/directions.pt \
  --output experiments/dino_adapter/image_tokens_step0 \
  --device cuda:0
```

`directions.pt` строится прежней командой `directions` по сохранённым train-парам;
она уже содержит и DINO-направления, и направления в h для каждого блока.
Здесь alpha — множитель сырой средней разности активаций, без нормализации по RMS
и без калибровки по адаптеру. Поэтому равные alpha в двух конфигурациях не означают
равную силу изменения атрибута. Используйте sweep, визуальную оценку и метрики.
Положительный alpha удаляет очки, отрицательный добавляет. `alpha=0` — контроль.

Для оценки сохранённых изображений:

```bash
python -m src.dino_adapter --config configs/image_tokens.json evaluate \
  --directions experiments/dino_adapter/directions.pt \
  --results experiments/dino_adapter/image_tokens_step0 \
  --device cuda:0
```

`metrics.json` содержит те же прокси-метрики, что и для адаптера. Автоматические
matched-effect пары формируются только внутри режима `adapter_comparison`;
результаты этого самостоятельного режима сравнивайте с ним по alpha-кривым.
