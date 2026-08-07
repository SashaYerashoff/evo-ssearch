# EVA 0.8.5 — профиль порта: 8 регулируемых каналов

Статус: Ventspils maritime client release candidate. Целевая машина: RTX 4070
Super 12 GB, Intel Core i9 14-го поколения, 64 GB DDRAM, Ubuntu 24.04.

## Размещение нагрузки

- RTX 4070S: Qwen3-VL-4B в vLLM и закреплённый SigLIP2 base для независимой
  семантической индексации. vLLM ограничен своим memory envelope; SigLIP2
  работает в том же CUDA device отдельными микробатчами. OpenAI CLIP сохранён
  только для сравнения и чтения legacy-индекса.
- Intel QSV: аппаратный decode восьми потоков, если драйвер и источник
  поддерживают его.
- CPU/iGPU: плотный CV около 4 fps/channel. CPU fallback для SigLIP2 остаётся
  аварийным профилем, а не штатным восьмиканальным размещением.
- CPU/RAM: отдельный llama.cpp 9B endpoint для L3 deep review, concurrency 1.
  Он не выгружает и не замещает live 4B.

Рекомендуемый стартовый vLLM envelope: 32k context, 4 concurrent sequences,
FP8 KV, `max_num_batched_tokens=4096`, image cap около 100352 pixels и
GPU-memory utilization, удерживающий процесс примерно в 9.8–10.5 GB. Это
стартовые пределы, а не обещание пропускной способности: окончательные числа
принимаются только по портовому soak.

## Неизменяемый индекс и регулируемый VLM

Для каждого включённого канала один выбранный CV-кадр в секунду получает CLIP
embedding и сохраняется в PostgreSQL как `source=semantic_snapshot` независимо
от motion, probes, alert и решения отправлять кадр в VLM. Потеря cadence-slot
явно считается gap; динамическое прореживание CLIP запрещено.
Операторский `All evidence` включает эти записи. Непрерывный источник ищется
по часовым channel-shards без newest-first лимита в 20 000 строк; в `/ready`
видны `observed_hz`, `staleness_seconds`, `wall_gap_slots` и
`source_gap_slots` по каждому каналу.
Direct USB/V4L2 analytics читает устройство одним ffmpeg-процессом; двойной
V4L2 → multipart MJPEG → ffmpeg loopback не используется.

VLM видит подмножество этого непрерывного ряда:

| Режим | Каденс кадра | Deadline | Min / target / max |
|---|---:|---:|---:|
| quiet | 10 s | 120 s | 6 / 8 / 8 |
| watch | 5 s | 90 s | 6 / 8 / 10 |
| active | 2.5 s | 60 s | 8 / 12 / 12 |
| burst | 1 s | 30 s | 10 / 16 / 16 |
| degraded | 15 s | 120 s | 4 / 6 / 6 |

Во всех режимах accumulator hard cap равен 16. Глобальный бюджет учитывает
одновременно токены и slot-seconds: steady state — шесть эталонных L0
requests/minute, burst может занять ещё два и затем гасит долг. Agent,
alert/describe и rollup имеют один защищённый inference slot; L0 занимает его,
только пока защищённой очереди нет.

## L1–L3

- L1 (15 минут) и L2 (1 час) используют agent profile и общую защищённую
  очередь.
- L3 (8 часов) может использовать отдельный 9B endpoint. Оператор задаёт
  timezone, дни и начало/конец тихого окна через
  `GET/POST /luxriot/rollups/l3-schedule`.
- Само попадание во временное окно недостаточно: gate также требует низкой
  activity, отсутствия свежих alerts и ограниченного L0 debt/inflight.
- Если окно занято реальным событием, L3 откладывается в пределах
  `max_deferral_seconds`. Никакого предположения «ночью тихо» нет.
- L3 выдаёт только review/proposals. Он не меняет probes, thresholds, alerts,
  sampling или live routine автоматически.

## Стартовые переменные

```dotenv
EVOSSEARCH_LUXRIOT_ATTENTION_SCHEDULER_ENABLED=true
EVOSSEARCH_LUXRIOT_ATTENTION_EPISODE_DISPATCH_ENABLED=false
EVOSSEARCH_LUXRIOT_ATTENTION_EMBED_ALL_CHANNELS=true
EVOSSEARCH_LUXRIOT_ATTENTION_EMBEDDING_CADENCE_MS=1000
EVOSSEARCH_LUXRIOT_ATTENTION_REQUESTS_PER_MINUTE=6
EVOSSEARCH_LUXRIOT_SUMMARY_MAX_BATCH_FRAMES=16

EVOSSEARCH_LIVE_CLIP_BATCH_SIZE=8
EVOSSEARCH_LIVE_CLIP_BATCH_WAIT_MS=75
EVOSSEARCH_LIVE_CLIP_BATCH_QUEUE_CAPACITY=128
EVOSSEARCH_LIVE_CLIP_BATCH_TIMEOUT_SEC=15
EVOSSEARCH_LUXRIOT_CLIP_ASYNC_ENABLED=true
EVOSSEARCH_LUXRIOT_CLIP_ASYNC_WORKERS=8
EVOSSEARCH_LUXRIOT_CLIP_ASYNC_QUEUE_CAPACITY=64

EVOSSEARCH_SEMANTIC_SNAPSHOT_ARCHIVE_ENABLED=true
EVOSSEARCH_SEMANTIC_SNAPSHOT_ARCHIVE_QUEUE=512
EVOSSEARCH_SEMANTIC_SNAPSHOT_ARCHIVE_BATCH_SIZE=32

EVOSSEARCH_LUXRIOT_ROLLUP_L3_DEEP_ENABLED=true
EVOSSEARCH_LUXRIOT_ROLLUP_L3_QUIET_WINDOW_ENABLED=false
EVOSSEARCH_LUXRIOT_ROLLUP_L3_QUIET_WINDOW_TIMEZONE=Europe/Riga
```

Локальный 9B endpoint устанавливается и проверяется заранее, но deep L3
остаётся fail-closed, пока оператор не сохранит разрешённое тихое окно.
Закрытый gate откладывает deep review и не забирает ресурс у live-потока.

React является штатным UI этой клиентской ветки; `/?ui=legacy` остаётся
аварийным fallback без изменения конфигурации сервиса.

## Ёмкость архива

Восемь каналов дают 691 200 semantic rows/day. Лимит 5 млн строк — меньше
7.3 суток ещё до alerts и VLM summaries. Base64 thumbnails обычно дороже
512-dimensional float32 vector, поэтому перед длинным портовым retention
обязательны фактический замер GB/day на NVMe и настройка row/thumbnail
retention. Непрерывность в выбранном retention окне сохраняется; экономить
место скрытым снижением 1 Hz нельзя.

## Acceptance gates

После старта проверить `/luxriot/streams`:

- `semantic_snapshot_archive.counters.persisted_total` растёт примерно на
  число включённых каналов в секунду;
- `gap_total`, `dropped_total` и `failure_total` остаются нулевыми;
- `clip_microbatcher.average_batch_size` растёт при нескольких каналах, queue
  не упирается в capacity;
- `attention.l0_cost_budget.burst_debt_l0` возвращается к нулю после bursts;
- agent-turn начинает генерацию, не дожидаясь освобождения всей L0 очереди;
- L0 deadlines и 16-frame cap не нарушаются;
- NVMe write latency, PostgreSQL size, CPU temperature и GPU VRAM остаются
  стабильными минимум два часа смешанного quiet/motion soak.

Если semantic archive даёт gaps, это блокер портового запуска. Если не
укладывается только VLM, сначала увеличивается quiet/watch interval или
снижается visual token cap; CLIP 1 Hz не трогается.
