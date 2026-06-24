# EVA AI UI User Guide

Дата: 2026-06  
Ветка: `feature/secure-50-channel-foundation`  
Статус: практический guide по текущему beta UI для пилотного клиентского деплоя

## 1. Общая логика интерфейса

EVA AI - единая панель для поиска по архиву кадров, live video descriptions, CLIP-проб, агента и администрирования. Основные вкладки сверху:

- `Archive` - поиск и просмотр сохраненных кадров из архива.
- `Video` - live-превью Luxriot-каналов, запуск video descriptions, просмотр VLM feed.
- `Monitoring` - создание, настройка и запуск probes.
- `Agent` - агентный чат с инструментами поиска, анализа каналов, отчетов и настройки probes.

Доступ к вкладкам зависит от роли пользователя. Если у пользователя нет нужного права, вкладка или часть настроек скрывается. Каналы также фильтруются по `allowedChannelIds`: пользователь видит только свои каналы, а `*` означает доступ ко всем каналам.

Важно для текущего клиентского деплоя: `Offline Video Analysis` и `Probe Snap` временно скрыты в интерфейсе и не считаются обещанными клиенту функциями. Archive Research -> описание найденной картинки через LLM остается рабочим и не связано с этим скрытием.

## 2. Вход, роли и права

Вход выполняется через форму `Sign in`: username + password. После входа система работает через server-side session cookie и CSRF для изменяющих запросов.

Текущие роли:

- `admin` - полный доступ: настройки, пользователи, аудит, модели, probes, capture, export, все каналы.
- `engineer` - настройка моделей, prompts, probes, capture, диагностика и просмотр настроек.
- `operator` - рабочее использование: смотреть потоки/детекции, пользоваться агентом, запускать probes, создавать bookmarks.
- `viewer` - read-only просмотр доступных каналов, детекций и отчетов.

Канальные права задаются отдельно от роли. Пользователь может быть `operator` или `viewer` на все каналы или только на выбранный список. Если агент или архив "не видит" канал, сначала проверьте права пользователя и список каналов в Admin -> Users.

## 3. Archive Research

Вкладка `Archive` предназначена для поиска по сохраненному frame archive. В архив попадают:

- `Probe hit` - реальные срабатывания probes.
- `Video description` - кадры, сохраненные из batch видео-описания.
- `VLM alert` - кадры, привязанные к alert из video description.

### 3.1 Фильтры Frame Archive

Слева в `Frame Archive`:

- `Stream` - ограничить поиск конкретным каналом.
- `Probe` - выбрать конкретную probe или элемент архива.
- `Source` - `All frames`, `Probe hits`, `Video descriptions`, `VLM alerts`.
- `Time range` - быстрый период: 1h, 6h, 24h, 3d, 7d, all time.
- `From` / `To` - точное окно времени через календарные поля.
- `Rows` - сколько архивных записей загружать кнопкой `Load Archive`.

Если заданы `From`/`To`, они используются как точное окно времени. Это удобно для разбора инцидента: сначала сузить период, потом искать текстом или картинкой.

### 3.2 Семантический текстовый поиск

В `Text Query` пользователь пишет естественный запрос, например:

- "человек у входа ночью"
- "красная машина возле шлагбаума"
- "коробки в офисе"

Нажмите `Search`. Поиск применяет текущие фильтры stream/source/time автоматически. Результаты появляются в центральной области `Search Results`, а выбранный результат раскрывается справа в `Inspector`.

### 3.3 Image Search

В `Image Query` можно загрузить эталонную картинку и нажать `Search by Image`. Это полезно для поиска похожего объекта, одежды, фрагмента сцены или референса из внешнего файла.

Image search также использует текущие фильтры stream/source/time.

### 3.4 Min match slider

`Min match` не меняет сам запрос к backend. Он фильтрует уже полученный batch результатов в UI.

Шкала адаптивная: после каждого нового набора результатов UI берет минимальный и максимальный score в этом batch и растягивает ползунок на диапазон 0-100%. Поэтому:

- `All` показывает все результаты.
- `>= ...` скрывает элементы ниже текущего порога.
- При новом запросе сохраняется позиция ползунка, но диапазон пересчитывается под новые scores.

Это особенно важно, потому что текстовые и image-search scores могут жить в разных диапазонах. Для картинок нормальны высокие значения вроде `0.7-0.99`, для текстовых запросов могут быть рабочие значения около `0.2-0.35`.

### 3.5 Что смотреть в карточке результата

В карточке и `Inspector` отображаются:

- `Name` - имя probe или archive item.
- `Source` - `Probe hit`, `Video description`, `VLM alert`.
- `Time` - время кадра.
- `Stream` - канал.
- `Severity` - severity события.
- `Match` - similarity для текущего поиска.
- `CLIP` / `Fusion` / `DINO` badge - какой режим поиска использовался.

Для `Probe hit` дополнительно показываются scores:

- `P` - positive score.
- `N` - negative score.
- `M` - margin.

Эти значения нужны для настройки noisy probes: сначала запускаем широкую пробу, копим срабатывания, смотрим реальные `P/N/M`, затем подкручиваем `Positive`, `Negative`, `Margin` и bookmark gate.

### 3.6 Preview и describe detection image

В `Inspector` можно открыть превью найденного кадра и запросить описание картинки через LLM (`Describe with LM`). Эта функция должна оставаться включенной в клиентском деплое.

Важно: это отдельный сценарий от скрытого `Offline Video Analysis`. Даже если Offline Video Analysis скрыт, описание конкретной картинки из Archive Research работает через `/describe_image`.

## 4. Video

Вкладка `Video` управляет live-потоками Luxriot Evo и video descriptions.

### 4.1 Live Stream Control

Слева:

- `Channel` - выбор Luxriot-канала.
- `Reload` - перечитать список каналов.
- `Batch` - сколько кадров отправлять в один VLM-запрос.
- `Every` - интервал между снапшотами в секундах.
- `Live model` - модель или профиль для live summaries.
- `Start summaries` / `Stop summaries` - запуск/остановка video descriptions.
- `Flush now` - принудительно отправить накопленный batch.
- `System prompt settings` - prompts для live summaries, L1-L3 rollups и JSON alerts.

В `System prompt settings` верхнее текстовое поле - это редактируемая часть prompt для выбранного канала и уровня. Блок `Effective prompt layers` показывает дополнительные слои, которые EVA AI добавляет автоматически:

- backend-инструкции для L1-L3 rollups: формат секций, `Alert Ledger`, `MEMORY_UPDATE_JSON`, запрет схлопывать alerts/deviations в рутину;
- активная память канала из прошлых L1-L3 summaries, если она уже накоплена;
- для L0/live summaries - память канала и JSON-инструкции для bookmarks, если bookmarks включены.

Это ожидаемое поведение: UI prompt задает операторскую задачу, а backend-слой удерживает стабильный формат, память и правила безопасности. Если в L1-L3 видны секции `Routine Baseline`, `Preserved Deviations`, `Alert Ledger`, `Alert Tuning Notes`, значит используется новый memory-aware rollup pipeline.

Для масштабного пилота рекомендованный путь - `Live model: Auto balance`, чтобы EVA AI распределяла каналы по настроенным VLM профилям. Ручной выбор `vlm-a1`, `vlm-a0`, `vlm-b1`, `vlm-b0` полезен для диагностики конкретной GPU/ноды.

### 4.2 Stream context

Справа от live preview отображается `Selected Stream`:

- имя канала;
- состояние `idle/running`;
- channel id;
- preview resolution;
- cadence;
- batch;
- live model;
- summary queue;
- probe capture;
- last preview.

Здесь же есть быстрые кнопки `Start summaries` и `Flush`.

Если summary queue растет, а VLM feed не пополняется, проверьте конкретный VLM endpoint и лимиты vLLM. Для текущей схемы live batch из 12 кадров требует, чтобы vLLM был поднят с image limit не ниже 12, практически - `--limit-mm-per-prompt.image 16`.

### 4.3 VLM Feed и L0-L3

`VLM Feed` показывает live summaries и rollups:

- `L0 / Live` - прямые batch summaries.
- `L1 / Minutes` - rollup за короткие интервалы.
- `L2 / Hours` - часовые сводки.
- `L3 / Days` - дневные сводки.

Фильтры:

- `Channel` - по какому каналу смотреть summaries.
- `History` - 6h, 24h, 3d, 7d, 30d, all history.
- `Depth` - L0/L1/L2/L3.

Кнопки:

- `Refresh` - обновить список.
- `Live` - следовать за новым feed.
- `Jump to latest` - перейти к последнему элементу, если пользователь прокрутил вверх.
- `Collapse all` - свернуть все entries.
- `Back` - вернуться после drill-down.

Свернутые записи показывают alert indicators: количество alerts по severity внутри summary-отрезка. В раскрытой записи доступны текст summary и действия:

- `Expand` - раскрыть/свернуть.
- `Copy` - скопировать текст.
- `Export` - выгрузить summary.
- `Bookmark` - создать bookmark в Luxriot, если у пользователя есть право.

Для L1-L3 поле `Alert Ledger` должно сохранять source alerts из L0 даже тогда, когда окно в целом рутинное. Например, `normal 2` в свернутой строке означает, что два source alerts попали в окно; текст rollup может пометить их как routine/no action, но не должен полностью скрывать их.

Один L0 batch может содержать несколько независимых alerts. Если в одном окне одновременно видны разные операторские триггеры, VLM должен вернуть один `ALERTS_JSON` с массивом `alerts`, где каждый объект описывает отдельное событие и имеет свою `severity`. EVA AI по умолчанию обрабатывает до 8 alerts на batch; лимит можно поднять/снизить через `EVOSSEARCH_LUXRIOT_ALERTS_MAX_PER_BATCH`.

Основной контракт - `ALERTS_JSON`. Для устойчивости EVA AI также распознает явные prose-строки в секции `Alerts`, например `Info Level: ...` и `Warning Level: ...`; `Warning` нормализуется в canonical severity `low`. Такой fallback нужен только как страховка, нормальный prompt должен требовать JSON.

VLM alerts и часть summary frames также попадают в Archive Research, чтобы их можно было найти позже как `Video description` или `VLM alert`.

### 4.4 Channel Runtime

В правой нижней панели `Channel Runtime` отображаются активные video/probe sessions. Полезные действия:

- `Refresh` - обновить runtime состояние.
- `Stop video` - остановить все video-description sessions.
- `Pause probes` - поставить probes на паузу.

После рестарта `eva-ai` live summaries не восстанавливаются автоматически: их надо снова запустить из UI или скриптом.

### 4.5 Временно скрытая Offline Video Analysis

`Offline Video Analysis` временно скрыт для клиентского деплоя. Причина: сценарий отдельного upload/server-path анализа не был обещан клиенту и требует дополнительной проверки. Не показываем его оператору, чтобы не смешивать рабочий live workflow с экспериментальной функцией.

## 5. Monitoring / Probes

Вкладка `Monitoring` отвечает за сохраненные probes, их запуск и просмотр последних срабатываний.

### 5.1 Saved Probes и Probe Board

Слева:

- `Refresh list` - перечитать probes.
- `+ New Probe` - создать новую probe.
- `Run benchmark` - оценить CLIP throughput для sizing.

В центре `Probe Board` показывает карточки probes:

- состояние `running`, `paused`, `idle`, `disabled`;
- канал;
- последнее срабатывание;
- scores `P/N/M`;
- bookmark gate state;
- быстрые actions: открыть, run, enable/disable, delete.

Справа `Selected Probe` показывает подробности выбранной probe и прямые действия:

- `Probe settings` - открыть editor.
- `Run probe` / `Stop probe` - запуск/остановка.
- `Delete Probe` - удалить.

`Latest Detections` показывает последние hits с thumbnails, временем и scores `P/N/M`.

### 5.2 Создание и редактирование probe

В `Probe Settings`:

- `Channel` - канал, к которому привязана probe.
- `Start Stream` - запуск probe capture для канала.
- `Probe name` - понятное имя.
- `Enabled` - активна ли probe.
- `Make bookmarks` - создавать bookmarks в Luxriot при hits.
- `Severity` - severity для bookmark.

Text probe:

- `Positive` - описание того, что надо ловить.
- `Negative` - описание похожего, но нежелательного шума.
- `Positive` threshold - минимальный positive score.
- `Margin` - насколько positive должен быть выше negative.

Image probe:

- загрузить эталонную картинку;
- включить `Enabled`;
- задать `Minimal match`.

Правило настройки: positive формулируем как наблюдаемую визуальную сцену, negative - как похожие ложные срабатывания. Чем больше реальных hits собрано, тем точнее можно выставлять thresholds.

### 5.3 ROI

ROI ограничивает область кадра для matching:

- `ROI OFF/ON` - включить или выключить область.
- `Clear ROI` - очистить область.

ROI полезен, когда объект важен только в конкретной зоне кадра: дверь, проход, касса, въезд, полка.

### 5.4 Cast Probe

`Cast` позволяет применить одну probe к нескольким каналам:

- `Current`, `All`, `None` - быстрый выбор каналов.
- `Conflict`:
  - `Skip matching` - не трогать существующие похожие probes.
  - `Update matching` - обновить существующие.
  - `Create copies` - создать копии.
- `Enabled` - новые/обновленные probes сразу активны.
- `Copy ROI` - копировать ROI.
- `Start streams` - сразу стартовать probe capture.

Это основной способ быстро размножить одинаковую probe на 30-50 каналов.

### 5.5 Временно скрытый Probe Snap

`Probe Snap` временно скрыт для клиентского деплоя. Причина: snapshot должен быть синхронизирован с актуальным frame buffer, иначе оператор может получить не тот кадр или "preview frame is not ready yet".

Ручная настройка image probe через `Choose Image` остается доступной.

## 6. Agent

Вкладка `Agent` - чат с инструментами EVA AI. Агент работает в рамках прав текущего пользователя и видит только разрешенные каналы.

Основные возможности:

- поиск по архиву;
- последние detections;
- summary по detections;
- список каналов;
- список probes и их статусы;
- описание текущего кадра с канала;
- получение video summaries за период;
- генерация отчетов;
- создание bookmarks;
- preview/apply для изменения probes и prompt settings;
- survey каналов перед настройкой probes.

В UI есть готовые chips:

- `Image`
- `Latest detections`
- `Probe status`
- `Daily report`
- `Archive search`
- `Describe frame`

### 6.1 Как правильно задавать период и канал

Лучшие запросы агенту:

- "Покажи video summaries по каналу 105 за последние 6 часов."
- "Найди в архиве по каналам 105,109 людей у входа с 22:00 до 02:00."
- "Сделай краткий отчет по L0/L1 summaries канала TVT за ночь."
- "Проверь последние detections по probe `person-at-door` за 24 часа."

Если канал не указан, агент должен сначала выяснить доступные/активные каналы. Для большого числа каналов он не должен молча делать полный обход: он должен предложить кандидатов и попросить подтверждение на полное исследование, потому что такой запрос может занять много времени и несколько turns.

### 6.2 Когда просить Agent, а когда Archive

Используйте `Archive`, если оператор сам настраивает фильтры и хочет глазами просмотреть кадры.

Используйте `Agent`, если нужен текстовый отчет, cross-channel reasoning или "найди и объясни" по нескольким источникам.

Для настройки probes агент полезен как помощник: попросить оценить noisy hits, предложить positive/negative формулировки и margin. Изменения probes должны идти через preview/apply, чтобы оператор видел, что именно будет изменено.

## 7. Settings, Admin, Audit

Settings открываются через иконку шестеренки. Видимость секций зависит от прав.

### 7.1 Server

Базовые параметры Flask control-plane:

- host;
- port;
- debug mode.

В прод/пилоте debug должен быть выключен.

### 7.2 Search

Настройки default/min/max результатов для archive и agent-assisted search.

### 7.3 Models

Настройки embedder backend:

- `CLIP`;
- `DINO`;
- `Fusion`.

Для текущего клиентского деплоя рабочий и рекомендуемый режим - CLIP. DINO/Fusion считаются экспериментальными и не должны продаваться как основная клиентская функция без отдельной проверки.

Также здесь задаются CLIP/SigLIP model, batch size и thumbnail quality.

### 7.4 Advanced

Дополнительные controls:

- max comment length;
- max file size;
- rerank;
- segment embeddings;
- storage/index internals.

Indexed folder controls скрыты из основного workflow, потому что в этом деплое основной путь - archive + PostgreSQL-backed frame archive, а не ручной indexed-folder сценарий.

### 7.5 Luxriot

Здесь задаются:

- Luxriot Evo base URL, username, password;
- default channel id;
- snapshot interval;
- snapshot max edge;
- max buffer frames;
- description retention;
- description cap per channel;
- auto bookmark alerts;
- probe bookmark cooldown/dedupe/gate параметры;
- severity mapping.

### 7.6 Archive Capacity / Retention

В `Archive Capacity` задаются:

- включена ли retention policy;
- сколько дней хранить frame rows;
- сколько дней хранить DB previews/thumbnails;
- max frame records;
- плановое количество каналов;
- frames per batch;
- средний JPEG size;
- probe rows per channel per day.

UI показывает оценку емкости. Эти настройки нужны до запуска long-running пилота, иначе previews и archive rows могут быстро занять диск.

### 7.7 Users & Sessions

Доступно администраторам (`users:manage`).

Можно:

- создать пользователя;
- задать display name;
- задать/reset password;
- выбрать роли;
- выдать каналы через picker или `*`;
- включить/выключить пользователя;
- revoke sessions;
- посмотреть текущие sessions.

Для операторов на клиентском стенде предпочтительно создавать named accounts, не общий admin.

### 7.8 Audit

Доступно пользователям с `audit:view`, обычно admin.

Фильтры:

- result: success/failure/denied;
- action;
- actor id;
- channel;
- request id;
- limit.

Audit нужен для принципа "нельзя коснуться системы, не оставив отпечаток": логируются входы, отказы, изменения users/roles/channels, probes, prompts, bookmarks, agent tool actions и доступ к чувствительным операциям.

### 7.9 Environment

Показывает и позволяет сохранять `EVOSSEARCH_*` overrides в `.env`. Изменения env обычно требуют restart сервиса `eva-ai`, чтобы полностью примениться.

Секреты должны обрабатываться осторожно: не показывать пароли клиенту на экране без необходимости.

## 8. Быстрые рабочие сценарии

### Разобрать инцидент по времени

1. Открыть `Archive`.
2. Выбрать stream или `All streams`.
3. Указать `From` / `To`.
4. Выбрать source: `All frames` или конкретно `VLM alerts`.
5. Выполнить text search.
6. Подвинуть `Min match`, чтобы оставить только сильные совпадения.
7. Открыть нужные кадры в `Inspector`.
8. При необходимости нажать `Describe with LM`.

### Запустить video descriptions на канале

1. Открыть `Video`.
2. Выбрать `Channel`.
3. Выставить `Batch` и `Every`.
4. Выбрать `Auto balance` или конкретный `vlm-*` профиль.
5. Нажать `Start summaries`.
6. Проверить `Selected Stream`: модель, queue, last preview.
7. Через 30-60 секунд проверить `VLM Feed`.

### Настроить noisy text probe

1. Открыть `Monitoring`.
2. Создать probe или открыть существующую.
3. Добавить positive и negative examples.
4. Начать с мягких thresholds.
5. Запустить probe и накопить hits.
6. Смотреть `P/N/M` в `Latest Detections` и `Archive`.
7. Увеличивать `Positive` и `Margin`, уточнять negative examples.
8. Включить bookmarks только после приемлемого уровня шума.

### Размножить probe на много каналов

1. Настроить probe на одном канале.
2. Нажать `Cast`.
3. Выбрать каналы.
4. Выбрать conflict policy.
5. При необходимости включить `Start streams`.
6. Применить и проверить `Probe Board`.

### Попросить агента проверить событие по видео-описаниям

Формулируйте запрос как: канал или группа каналов, событие, интервал.

Примеры:

- `Проверь канал Entrance 2: когда у двери впервые появился мусор или жидкость сегодня утром? Дай кадры.`
- `Был ли почтальон у входа с 09:00 до 12:00?`
- `Что происходило на центральной площади сегодня ночью? Дай только заметные происшествия и визуальные доказательства.`
- `Сравни: это похожий BMW дрифтил на площади и позже врезался в забор?`

Ожидаемое поведение агента:

- сначала указать фактическое покрытие периода;
- для длинных периодов начинать с L2/L1 как обзорной карты;
- drill-down в L0 делать только по кандидатным окнам;
- визуальное подтверждение давать только по `VLM summary` / `VLM alert` кадрам с thumbnails;
- если каналов много, работать чанками и явно перечислять непроверенные каналы;
- не обвинять и не делать скрытых выводов. Например, `непривитая собака` должна стать `собака без видимой бирки/метки`, а `противозаконное` - `видимый инцидент для проверки оператором`.

## 9. Ограничения текущего деплоя

- `Offline Video Analysis` скрыт и не обещан клиенту.
- `Probe Snap` скрыт и не обещан клиенту.
- DINO/Fusion не являются основным клиентским режимом без отдельной проверки.
- Live summaries после restart `eva-ai` надо запускать заново.
- Auto balance распределяет каналы по настроенным VLM profiles; при диагностике проблем используйте ручной выбор профиля.
- Archive search показывает то, что уже было сохранено: если summaries/probes не работали или retention очистила данные, искать будет нечего.

## 10. Change Control

Этот guide должен обновляться при каждом изменении UI/UX или новой клиентской функции.

Минимальные правила:

- Если меняются вкладки, кнопки, labels или доступность controls - обновить соответствующий раздел.
- Если feature скрыта/включена для деплоя - явно пометить статус и причину.
- Если меняются роли/permissions - обновить раздел входа и Settings/Admin.
- Если меняется archive schema/source labels - обновить Archive Research.
- Если меняется VLM/probe workflow - обновить Video и Monitoring.
- Если агент получает новые tools или ограничения - обновить Agent.
- В release notes указывать, какой user guide соответствует сборке.
