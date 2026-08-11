# Полная миграция EVA AI UI на React — чек-лист

Обновлено: 2026-07-27

## Зафиксированный результат аудита

**Итог:** полного функционального дублирования старого UI в React сейчас нет.
React является рабочим прототипом с частичным паритетом. Переключать production
entrypoint на React до закрытия P0-пунктов этого файла нельзя.

Этот вывод относится к следующему неизменяемому снимку исходников:

| Объект проверки | SHA-256 / значение |
|---|---|
| `templates/index.html` | `8d426effe94ef7a7b50588923b5b05859c6ac5b05b70600ae0831a8be2a1f934` |
| `static/js/app.js` | `67769fc1eabff44aeec9af332ef4c2361e40aff4eebf1d8527f09cd5adee8fd4` |
| `static/css/app.css` | `4195876bcfc64b2f6aea0044aec269c9288bd74160cececf0c1d3a7f625045b1` |
| `oldapp.py` | `5076f22e4f43c68423d37a61f62597a792ac08abdefe10dc6b15ecceac4bb69f` |
| `security/http_auth.py` | `adba2f607467b2d237a6b22feadb8510c0c18f62c6adf05332962554ea44e35d` |
| `security/postgres_identity.py` | `f07bf1d2cbac6dc5d6cacdb98e8604efa69defaeac6fd442771125289c57a97a` |
| manifest 42 файлов `react-ui/src`¹ | `e4174cc6f2e71198c0deb7108e4e2bcb8cdcea59c79841c680342a03f486ac23` |
| `react-ui/vite.config.ts` | `90aa773fae8b6830c5ec85681da42437bf84cb9c7c96cfeb6fa12c869ddfe8fa` |
| `react-ui/package.json` | `61c65095b4f114c88bb83f7fb0860b307c29524a4231211664ca059988c9d6b7` |
| `react-ui/package-lock.json` | `2636207d0de120d3ec4f5ce9fd5720aef00eea5462f809216c0298dbbf4efa16` |

¹ SHA-256 от UTF-8 manifest со строками
`<sha256 файла>  <relative/path>`, отсортированными по полному пути, с
завершающим LF.

Инвентарь снимка: 353 уникальных legacy DOM ID, 495 именованных legacy JS
functions и 42 файла React source. Для каждой группы проверялись:
`видимый control → legacy handler → endpoint/method → response schema →
permission/channel scope → React consumer → acceptance`.

Статический route-аудит подтвердил: все endpoint/method pairs, которые сейчас
вызывает React, существуют во Flask. Проблемы находятся в consumer schemas,
неполных payload/filters, permission rendering и отсутствующих legacy flows.

Повторный аудит этого же снимка должен дать тот же результат. При изменении
любого хэша проверяются только затронутые строки матрицы и соответствующие
acceptance-сценарии.

Текущий security baseline из `security/permissions.py`:

| Роль | Permissions |
|---|---|
| `viewer` | `streams:view`, `detections:view`, `reports:view` |
| `operator` | viewer + `agent:use`, `probes:run`, `bookmarks:create` |
| `engineer` | viewer + `agent:use`, `probes:run`, `probes:manage`, `prompts:manage`, `models:manage`, `capture:manage`, `diagnostics:view`, `settings:view` |
| `admin` | все permissions, включая users/settings/audit/export |

Обязательные варианты channel scope для acceptance: `["*"]`, один channel,
несколько channels и пустой список. В текущем `.env` auth включён, а
`INDEXED_FOLDER`, `OFFLINE_VIDEO` и `PROBE_SNAP` выключены; это объясняет P2,
но не разрешает удалять соответствующий legacy-код без продуктового решения.

### Итоговая матрица паритета

| Область | Статус | Блокирующий остаток |
|---|---|---|
| Production-раздача | Нет | Flask продолжает отдавать legacy template; React build не включён в runtime/release |
| Shell/Auth | Частично | session-expiry E2E, degraded status и auth-disabled deployment |
| Archive | Частично | batch review, coverage/search mode и расширенные metadata |
| Video | Частично | отдельный feed channel, rollup UX, runtime и road mask |
| Monitoring | Частично | непрерывный run loop, точный runtime per probe, start stream, cast start-streams |
| EVA Agent | Частично | terminal event handling, reconnect и сохранённые tool traces |
| Settings/IAM/Audit | Частично | reset/enable actions и Settings form parity |
| Feature-flag функции | Решение не принято | indexed folder, offline video, probe snapshot, comments и segmentation |
| Автотесты React | Начато | 22 contract regression tests есть; component/browser E2E ещё отсутствуют |

### Проверенные API/consumer-контракты

| Контракт | Результат | Статус |
|---|---|---|
| `GET /audit/events` | camelCase backend response нормализуется в React model | Исправлено, test |
| `GET /settings/archive_capacity` | renderer читает `daily/retained/bytes` и `current.row_count` | Исправлено, test; live recalculation ещё открыт |
| `POST /auth/users/:id/revoke-sessions` | `revokedSessions` преобразуется в UI count | Исправлено, test |
| `GET /probes/status` | duration вычисляется из `[first,last]` | Исправлено, test |
| `GET /luxriot/session` | L0 History передаёт `from_ts` | Исправлено, test |
| Agent image-only | пустой text получает prompt `Describe this image.` | Исправлено, test |
| Archive list/search | общий serializer передаёт channel/source/probe/time/limit/sort; list передаёт offset | Исправлено, 3 tests |
| Agent SSE | incremental parser обрабатывает chunk boundaries, LF/CRLF, malformed frame и EOF remainder | Исправлено, test |
| Prompt settings без `bookmarks:create` | JSON-alert и bookmark-поля не отправляются; соответствующие controls скрыты | Исправлено, test |
| Probe settings без `bookmarks:create` | severity сохраняется; bookmark-поля удаляются; bookmarked probe становится read-only | Исправлено, test |
| Users wildcard channels | снятие одного channel при `*` создаёт «все доступные кроме выбранного», а не единственный ID | Исправлено, test |
| Auth current session | login/me возвращают `sessionId`; UI маркирует и защищает текущую session | Исправлено, backend + test |

### Видимые legacy-сценарии, которых нет в React

- Archive: batch review/filmstrip, переход в VLM feed, comments и
  segmentation/mask flows.
- Video: отдельный channel для live capture и просмотра feed, Jump to latest,
  Collapse all, drill-down/back, bookmark,
  copy/export, Channel Runtime, road mask и feature-flag Offline Video.
- Monitoring: непрерывный probe query loop, start/stop stream toggle,
  `Start streams` при Cast и feature-flag Probe Snapshot.
- Agent: восстановление tool cards при открытии session и stream reconnect.
- Users/Audit: отдельные reset/enable actions.

Legacy endpoints, вызываемые старым UI, но не вызываемые текущим React:

| Группа | Endpoints | Решение |
|---|---|---|
| Folder/index search | `/check_index`, `/index`, `/search`, `/search_by_image` | P2: перенести либо официально удалить |
| Comments | `/comments`, `/commented_images` | P2: перенести либо официально удалить |
| Segmentation | `/segment_from_point`, `/search_by_mask`, `/index_segments` | P2: перенести либо официально удалить |
| Offline media | `/video_understanding` | P2: перенести либо официально удалить |
| Road grounding | `/road/scene_overlay/:channel` | P1: перенести |
| Summary bookmark | `/luxriot/bookmark` | P0: вернуть действие в VLM feed |
| Probe draft/run | `/probes/query`, `/luxriot/snapshot/:channel/capture` | P0 run loop; snapshot остаётся P2 по feature flag |

## Цель

Полностью заменить старый интерфейс из `templates/index.html` и
`static/js/app.js` React-приложением из `react-ui`, сохранив рабочие сценарии,
права доступа, API-контракты и эксплуатационные свойства текущей системы.

Миграция считается завершённой не тогда, когда в React появились одноимённые
экраны, а когда:

- все поддерживаемые пользовательские сценарии имеют функциональный паритет;
- для каждой скрытой или устаревшей функции принято явное решение: перенести
  либо официально удалить;
- Flask в production отдаёт React-сборку;
- старый UI больше не нужен как runtime fallback;
- автоматические и ручные acceptance-тесты пройдены для всех ролей.

## Обозначения

- **P0** — блокирует переключение production на React.
- **P1** — требуется для полного функционального паритета.
- **P2** — legacy/feature-flag функция; требуется решение о переносе или удалении.
- `[x]` — реализация присутствует и статически подтверждена для снимка аудита
  выше. Это не заменяет production/E2E acceptance.
- `[ ]` — функция отсутствует, частична, сломана либо не подтверждена тестом.

## 1. Зафиксировать границы миграции

- [ ] **P0** Составить итоговую таблицу паритета `старый элемент → React-компонент
  → API → статус → acceptance-сценарий`.
- [ ] **P0** Зафиксировать набор поддерживаемых ролей, permissions и вариантов
  `allowedChannelIds`, на которых проверяется UI.
- [ ] **P0** Снять эталонные screenshots/video основных сценариев старого UI до
  начала удаления legacy-кода.
- [ ] **P0** Зафиксировать API-контракты, фактически используемые старым UI,
  включая успешные ответы, ошибки, пустые результаты и 401/403.
- [ ] **P0** Проверить интерфейс при реальной production-конфигурации, а не только
  с локальным `admin`.
- [ ] **P2** Для каждой feature-flag функции принять и записать решение:
  - `EVOSSEARCH_INDEXED_FOLDER_ENABLED`;
  - `EVOSSEARCH_OFFLINE_VIDEO_ENABLED`;
  - `EVOSSEARCH_PROBE_SNAP_ENABLED`;
  - segment/mask search;
  - folder comments и commented images.
- [ ] **P2** Если функция выводится из продукта, удалить её из старого UI, API
  документации и настроек отдельным изменением до финального cutover.

## 2. Production-интеграция React

- [x] Production-схема раздачи React определена: Flask отдаёт
  `react-ui/dist/index.html` и fingerprinted assets из `/ui-assets/`.
- [x] Воспроизводимый React build включён в USB bundle builder.
- [x] React source и готовый `dist` включены в patch/release bundle; production
  appliance не требует Node.js.
- [ ] **P0** Добавить SPA fallback для клиентских маршрутов, если будет
  использоваться routing.
- [x] React-раздача не перехватывает API, `/branding`, thumbnails и
  другие backend routes.
- [x] Настроены корректные cache headers:
  - `index.html` — `no-cache`;
  - fingerprinted assets — долгий immutable cache.
- [x] Production entrypoint и asset route проверяются backend contract tests, не
  только через Vite proxy на `:5173`.
- [x] Текущий Vite proxy включает все используемые React API-prefixes, включая
  `/audit`.
- [ ] **P2** При переносе legacy-функций добавить соответствующие proxy-prefixes:
  `/road`, `/video_understanding`, `/segment_from_point`, `/search_by_mask` и
  `/index_segments`.
- [x] Добавлен runtime feature flag `EVOSSEARCH_UI_MODE=legacy|react` и
  per-request pilot overrides `/?ui=react|legacy`.
- [x] Rollback не требует пересборки базы: legacy остаётся default, а React при
  отсутствии `dist` fail-safe возвращает legacy shell.
- [ ] **P0** Добавить health/smoke-проверку доступности React entrypoint и assets.
- [ ] **P1** Настроить source maps согласно политике production-сборки.
- [ ] **P1** Добавить страницу/состояние для ошибки загрузки JS bundle.

## 3. Общая оболочка приложения

- [x] Есть базовые разделы Home, Archive, Video и Monitoring.
- [x] Home по умолчанию работает как пассивная заставка без игрового HUD и
  управления: метаданные автоматически летят к глазу. Игра активируется пятью
  нажатиями по глазу; каждое нажатие запускает моргание. Скорость, частота
  появления и максимальное количество metadata frames увеличены вдвое.
- [x] Визуальная композиция Home переработана: глаз уменьшен и опущен ниже,
  мозг увеличен без разрыва optic-nerve/event координат. Добавлены анатомические
  слои, извилины, глубинная нейронная сетка, scanline/impulse-анимации и
  детализированная радужка с волокнами, кольцами и бликами.
- [x] Поток metadata frames привязан к `uptime_sec` текущего запуска backend:
  Home остаётся смонтированным при переходах между разделами, а летящие кадры,
  следующая генерация и состояние игры сохраняются и восстанавливаются после
  reload. Снимок от предыдущего запуска сервера автоматически отбрасывается.
- [x] Есть верхняя status bar, левое меню, Settings и Logout.
- [x] Есть Appearance editor с четырьмя палитрами, семантическими color
  overrides и contrast gate.
- [x] Appearance editor возвращён в Settings как встроенная вкладка; отдельная
  кнопка в TopBar и отдельная modal-точка входа удалены. Reset и Apply
  переиспользуют существующий редактор и сохранение предпочтений.
- [x] Footer вкладки Appearance приведён к общему стилю Settings без второго
  пустого footer: `Reset to defaults` и `Apply appearance` используют общую
  геометрию кнопок, а Apply сохраняет внешний вид и закрывает Settings.
- [x] Секции Appearance расположены одной вертикальной колонкой; демонстрационный
  блок `Live preview · EVA Deep` и декоративный sample фиксированной типографики
  удалены. Сохраняемый режим `Normal / Big 125%` вынесен в отдельную первую
  секцию; Big interface использует внутренний viewport, чтобы fixed-панели и
  modal overlays оставались в экране.
- [x] Есть отдельная боковая панель EVA Agent.
- [x] Docked EVA Agent использует фиксированные width-пресеты: Full HD держит
  `4/3` и никогда не опускается ниже трёх карточек, а 2K — `5/4/3`.
  Кнопка ширины переключает пресеты
  циклически; ручной drag при отпускании защёлкивается на ближайший пресет.
  Позиции панели вычисляются целыми grid-слотами (`card + gap`) по CSS viewport:
  при выдвижении Agent карточки сохраняют исходный размер и целыми колонками
  переходят на следующую строку вместо сжатия. В режиме `Big 125%` пресеты и
  drag-нормализация используют внутренний layout viewport (`window / 1.25`), а
  не физическую ширину экрана. После максимального пресета
  дальнейший drag расширяет панель поверх Archive, не сдвигая интерфейс ниже
  минимальных трёх колонок; при отпускании панель возвращается к пресету.
  Счётчик archive matches во время drag остаётся на месте и перемещается только
  после отпускания, одновременно с защёлкиванием панели на выбранный пресет.
- [x] Версия приложения загружается из `/health`; жёстко заданная версия из
  React удалена.
- [ ] **P0** Проверить статусы Luxriot, каналов, probes и Agent на реальных
  ответах, включая partial outage.
- [x] Статус Agent подключён к реальному `busy/streaming`.
- [x] Счётчик `/probes/list` подписан как configured probes, а не active.
- [ ] **P0** Не показывать «connected/active» только на основании одного
  успешного запроса при старте; определить TTL и degraded-состояние.
- [x] Глобальная обработка 401:
  - очистка текущей пользовательской сессии;
  - возврат на Login;
  - отсутствие бесконечных повторов запросов.
- [x] Глобальная обработка 403 показывает единое ограниченное сообщение, а
  недоступные разделы и основные mutation controls скрываются заранее.
- [ ] **P1** Сохранять выбранный раздел после reload, если это не нарушает
  security/session boundaries.
- [ ] **P1** Проверить keyboard navigation, focus trap в модальных окнах и
  восстановление focus после закрытия.
- [ ] **P1** Проверить responsive layout на целевых разрешениях оператора.
- [ ] **P1** Проверить режим отключения animations во всех новых компонентах.
- [ ] **P1** Добавить общий React error boundary с понятным сообщением и
  request/correlation ID, если он доступен.

## 4. Авторизация, IAM и ограничения доступа

- [x] Есть login через `/auth/login`.
- [x] Есть восстановление пользователя через `/auth/me`.
- [x] Есть logout через `/auth/logout`.
- [x] API client отправляет cookies и CSRF для mutating requests.
- [x] API client добавляет CSRF для JSON, multipart, PATCH и DELETE; helper
  покрыт unit test.
- [ ] **P0** Проверить истёкшую, отозванную и неактивную сессию.
- [x] Navigation, Settings, Video capture/prompts, Monitoring mutations,
  Agent model/skills и diagnostics controls отображаются согласно permissions.
- [x] `allowedChannelIds` применяется централизованно до передачи channels в
  Archive, Video, Monitoring, Agent context и Settings.
- [x] Wildcard `allowedChannelIds=["*"]` поддерживается и покрыт unit test.
- [x] Users/Audit/Environment/settings tabs скрываются без соответствующих прав;
  Audit дополнительно требует all-channel scope.
- [x] Top bar показывает текущего пользователя, Logout остаётся доступен в rail.
- [ ] **P1** Обработать deployment с отключённой auth, если такой режим остаётся
  поддерживаемым.

## 5. Archive

### 5.1 Загрузка и фильтры

- [x] При каждом открытии Archive по умолчанию развёрнут раздел `Filters`;
  Text query, Match и Image остаются свёрнутыми до выбора пользователем.
- [x] Есть нормализация `/detections/list` и `/detections/search_*` в общую
  модель `Detection`.
- [x] Есть фильтр по channel.
- [x] Есть фильтр по source.
- [x] Есть preset time range и custom date range.
- [x] Есть выбор количества строк.
- [x] Есть Load Archive.
- [x] Добавлен фильтр по probe; metadata загружается через
  `/detections/summary`, поэтому фильтр включает исторические archive probes,
  а не только текущий `/probes/list`.
- [x] Probe filter показывается только для source `probe`, как в legacy UI.
- [x] Добавлен Refresh Filters с обновлением channels и archive probes.
- [x] Реализована динамическая подгрузка при скролле через `limit`, `offset`,
  `total` и `has_more`; новые записи дописываются без замены уже загруженных.
- [x] Offset сбрасывается при изменении фильтра или limit.
- [x] Общее количество archive matches и количество загруженных записей
  показываются над результатами по центру в одной горизонтальной строке; оба
  числа оформлены одинаковым акцентом, размером и tabular-цифрами.
- [x] Используется стабильный key без `Math.random()`.
- [x] Request sequence не позволяет более старому list/text/image ответу
  перезаписать результат нового запроса.
- [x] Применённые фильтры сохраняются отдельно от draft; при изменениях подпись
  показывает `Filters changed — load to apply`, а автоподгрузка блокируется.
- [x] При сужении Archive выдвинутым Agent фильтры остаются в одной строке и
  переходят в горизонтально прокручиваемую ленту без видимого scrollbar;
  `Actions` и `Load archive` закреплены справа, высота панели не меняется.
  Верхняя строка вкладок использует то же поведение вместо сжатия и переноса.

### 5.2 Text/Image search

- [x] Есть text search через `/detections/search_text`.
- [x] Есть image search через `/detections/search_image`.
- [x] Есть сортировка по similarity/time.
- [x] Есть клиентский min-match filter.
- [x] `probe_id` передаётся в text и image search для source `probe`.
- [x] `since_ms` и `until_ms` передаются в text search.
- [x] Time preset/custom range передаётся в image search.
- [x] Одинаковый набор channel/source/probe/time/limit/sort
  фильтров во все три Archive-сценария.
- [x] Text search не запускается с пустым запросом.
- [ ] **P0** Показывать coverage, requested/used mode и fallback, если backend
  вернул эти данные.
- [ ] **P1** Добавить preview, filename и явное удаление выбранного query image.
- [ ] **P1** Проверить min-match при узком score range и результатах без score.
- [ ] **P1** Определить UI для embedder/search mode, если экспериментальные
  embedders остаются доступными оператору.

### 5.3 Карточки результатов

- [x] Превью Archive-карточек увеличены примерно вдвое с сохранением базовой
  пропорции `344×184`. Без открытого Agent grid всегда использует шесть равных колонок;
  размер карточек вместе с увеличенными gaps рассчитан на всю доступную ширину без пустого
  хвоста справа. В agent width-пресетах выбранное количество колонок также
  заполняет оставшуюся рабочую область без изменения размера карточек.
  Thumbnail жёстко обрезается внутри своей зоны, а название и metadata вынесены
  в отдельный фон с границей и отступом без наложения на изображение.
- [x] Есть preview, source, probe/name, channel, time, severity и match.
- [ ] **P0** Отображать данные без потерь для всех source:
  `vlm_summary`, `vlm_alert`, `probe`.
- [ ] **P0** Проверить `id/detection_id`, `image_path/path`,
  `recorded_at_ms/timestamp_ms`, `similarity/bookmark_gate.similarity`.
- [ ] **P1** Вернуть необходимые badges/metrics: CLIP, origin, P/N/M, bookmark
  state и другие подтверждённые поля старой карточки.
- [ ] **P1** Добавить понятное состояние кадра без thumbnail.
- [x] Итоговая подпись list/text/image/Agent результата отображается под grid.

### 5.4 Inspector и Archive Review

- [x] Есть базовый Inspector.
- [x] Есть Describe frame.
- [x] Есть Find similar.
- [x] Есть lightbox/zoom.
- [x] Inspector загружает full-resolution image через `/detections/image` и
  использует thumbnail только как fallback при ошибке.
- [ ] **P0** Добавить batch frame review для video summary results.
- [ ] **P0** Добавить Prev/Next frame и keyboard navigation.
- [ ] **P0** Добавить filmstrip кадров одного batch.
- [ ] **P0** Показывать user query, CLIP match, frame role, timestamp и summary.
- [ ] **P0** Добавить Open VLM feed с переходом к правильному channel/time/batch.
- [ ] **P1** Добавить Copy summary с feedback об успешном копировании.
- [ ] **P1** Определить, нужно ли сохранять LLM description как comment.
- [ ] **P2** Если folder comments остаются в продукте, перенести:
  - список комментариев;
  - добавление комментария;
  - commented images;
  - лимиты и ошибки.
- [ ] **P2** Если segmentation остаётся в продукте, перенести:
  - point segmentation;
  - mask preview;
  - search by mask;
  - index segments;
  - настройки threshold/min patches.

## 6. Video

### 6.1 Live stream control

- [x] Есть выбор channel, batch, interval и live model.
- [x] Есть Start/Stop summaries.
- [x] Есть Flush.
- [x] Есть live preview.
- [x] Есть prompt settings по слоям.
- [ ] **P0** Вернуть отдельный channel selector для VLM feed либо явно утвердить
  объединение с capture channel. В legacy эти контексты независимы.
- [x] Произвольный live prompt передаётся в `/luxriot/start_capture`.
- [x] Кнопка Reload channels повторно загружает `/luxriot/channels`, затем
  `/luxriot/streams`, и сохраняет выбранный channel, если он ещё доступен.
- [ ] **P0** Проверить все допустимые batch sizes и ограничения interval из
  backend config, не хранить их только как frontend-константы.
- [ ] **P0** Показывать реальное разрешение, cadence, model, queue, dropped
  frames, last frame и ошибки выбранного stream.
- [ ] **P0** Корректно различать video capture и analytics/probe capture.
- [ ] **P0** Обработать channel без свежего кадра, timeout и frozen signal.
- [ ] **P0** Синхронизировать control values с уже запущенным stream.
- [ ] **P1** Добавить подтверждение для опасных stop-all действий.
- [ ] **P1** Проверить prompt layers/effective prompt и server validation.
- [x] Prompt save без `bookmarks:create` не отправляет `json_alert_prompt`,
  `bookmark_enabled` и cooldown; renderer скрывает JSON-alert tab и bookmark
  controls.

### 6.2 VLM feed и rollups

- [x] На Archive, Stream Summaries и Probes шапка вкладок и фильтров закреплена
  сверху и не прокручивается вместе с рабочим содержимым. Вкладки разделов
  (`Stream review` / `Stream settings` и аналоги) находятся внутри `atp-tabpanel`
  над фильтрами. Внешний `atp-tabpanel` и `atp-tabrow` прозрачны: вкладки лежат
  на общем фоне страницы. `.atp-tabpanel-content` также не имеет общей заливки,
  рамки, blur или тени: локальный фон и границы принадлежат только фильтрам,
  полям, action-кнопкам и кликабельным `.atp-tab`. Вкладки оформлены как
  компактные полупрозрачные chips со скруглением со всех сторон; активная вкладка
  получает лёгкий accent tint и тонкую нижнюю метку вместо тяжёлого folder-tab.
  Все вкладки показывают только иконку и название раздела: выбранные каналы,
  фильтры, запросы, режимы, счётчики и другие динамические summary в ушки не выводятся.
  Под контролами активного `.atp-tabpanel-content` постоянно течёт едва заметная
  размытая «река света»: бесшовный поток покрывает всю ширину без пустых пауз,
  сильно растворяется маской у левого и правого края, не перехватывает ввод и
  отключается общим режимом reduced/no motion. Сам
  `.tool-tabs.with-leading` не имеет фоновой полосы или нижней границы.
- [x] Все плотные workspace-toolbar используют единый responsive-контракт:
  Archive Filters/Text Query, Stream Review, Stream Settings и Probes сохраняют
  одну высоту при сужении выдвинутым Agent. Центральные контролы прокручиваются
  горизонтально без видимого scrollbar, а `Actions` и главное действие остаются
  закреплены справа и не перекрывают фильтры. Когда лента действительно не
  помещается, обычное вертикальное колесо мыши прокручивает её горизонтально;
  на краях управление возвращается вертикальному scroll рабочей области.
- [x] `TopBar` не рисует отдельную залитую полосу: фон и backdrop blur убраны,
  нижняя разделительная линия отсутствует, через шапку виден общий фон приложения.
  Вторичный слоган `Smart Image Search and Understanding` удалён; бренд-блок
  содержит только логотип, `EVA AI` и версию приложения.
- [x] Dock агента не выглядит табличной сеткой: сплошная непрозрачная заливка и
  горизонтальные разделители шапки, toolbar, composer и input убраны. Оболочка
  использует полупрозрачную глубокую поверхность с одной границей у workspace,
  а ответы EVA читаются на локальном растворяющемся световом слое без рамки.
- [x] Full-screen агента использует непрозрачную поверхность, а workspace-контент
  под ним скрывается; нижний status console в этом режиме исключён из layout.
  Слева и справа сохранён системный gutter шириной edge-trigger + 12 px: через
  него виден общий neural background, но не карточки workspace, а «уши» главного
  меню и AGENT не перекрывают содержимое dock.
  Читаемость сохраняют только локальные полупрозрачные подложки под brand lockup
  и названием текущего раздела; они не образуют общую полосу. Отдельный radial
  background у Home также убран, чтобы граница его рабочей зоны не выглядела
  остаточной заливкой `TopBar`.
- [x] Нижний `status-console` оформлен по той же схеме: общая заливка, верхняя
  линия и backdrop blur убраны, а минимальная подложка применяется только к
  отдельным текстовым группам статуса.
- [x] Глобальный trigger меню заменён компактной edge-рейкой
  `.menu-rail-trigger`: вместо набора плохо различимых мини-иконок она показывает
  компактный знак из трёх горизонтальных линий и целиком открывает полноразмерное меню.
  Единственный экземпляр рейки рендерится через `LeftRail` во всех разделах, а
  tabbed-разделы резервируют под него пустую leading-колонку `ToolTabs` шириной
  `28px`. Рейка растягивается ровно по высоте `.atp-tabpanel-content`, поэтому её
  положение и размер не прыгают при переключении разделов и вкладок.
- [x] Переключение вкладок не пересоздаёт и не анимирует весь `atp-tabpanel`:
  геометрия шапки остаётся неподвижной, обновляется только содержимое панели.
- [x] На Archive прокручивается только внутренний блок результатов: шапка,
  фильтры и счётчик совпадений остаются на месте, а grid карточек и sentinel
  динамической подгрузки находятся в отдельном scroll-контейнере.
- [x] В `Stream settings` удалены дублирующие групповые заголовки `Stream source`,
  `Sampling and batching`, `Inference` и `Runtime`; сохранены только подписи полей.
  Группы также не имеют собственных фонов, рамок и скруглений — контролы лежат
  непосредственно на общем фоне панели.
- [x] Из `Stream review` удалена дублирующая кнопка `Edit settings`: переход к
  настройкам выполняется через соседнюю вкладку `Stream settings`.
- [x] Контролы `Stream settings` приведены к общей toolbar-геометрии: высота
  полей и кнопок `38px`, стандартные ширины Channel/Batch/Every/Model
  `320/92/84/220px`; поля больше не растягиваются по всей ширине экрана.
- [x] Toolbar actions получили единую визуальную иерархию: главное действие
  текущего таба остаётся акцентной кнопкой справа без постоянной яркой заливки
  и glow, вторичные команды собраны в один dropdown `Actions`, а
  `Collapse all / Expand all` объединены в stateful action.
- [x] Есть L0 feed и L1/L2/L3 rollups.
- [x] Есть manual refresh, live polling и history range для L1/L2/L3.
- [x] L0 History передаёт `from_ts` в `/luxriot/session`.
- [ ] **P0** Реализовать корректный live-follow:
  - остановка автоскролла при ручном просмотре;
  - Jump to latest;
  - явное состояние Live on/off.
- [ ] **P0** Добавить Collapse all и состояние отдельных summary/rollup.
- [ ] **P0** Реализовать drill-down/back между уровнями rollup.
- [ ] **P0** Добавить alert badges и корректную severity mapping.
- [ ] **P0** Добавить Bookmark для summary при наличии permission.
- [ ] **P1** Добавить Copy и Export для summary и rollup.
- [ ] **P1** Сохранять/показывать выбранный channel и временной контекст при
  переходе из Archive.
- [ ] **P1** Проверить отображение markdown и sanitization недоверенного текста.

### 6.3 Channel Runtime

- [x] React-компонент `ChannelRuntime` создан.
- [ ] **P0** Подключить `ChannelRuntime` к `VideoScreen`; сейчас он не
  рендерится.
- [ ] **P0** Реализовать Refresh runtime.
- [ ] **P0** Реализовать Stop video, Pause probes и Stop all для одного channel.
- [ ] **P0** Реализовать глобальные Stop video и Pause probes.
- [ ] **P0** Добавить View summaries с переключением выбранного channel.
- [ ] **P0** Показывать desired-but-missing streams и last errors.
- [ ] **P1** Проверить согласованность runtime после действий из Monitoring и
  Agent.

### 6.4 Road scene grounding

- [ ] **P1** Перенести Ground road mask.
- [ ] **P1** Показывать overlay, confidence, metadata, busy/error/empty states.
- [ ] **P1** Проверить permissions и ограничения channel.

### 6.5 Offline Video Analysis

- [ ] **P2** Принять решение: функция поддерживается или официально удаляется.
- [ ] **P2** Если поддерживается, перенести:
  - upload image/video и server path;
  - выбор model;
  - frame count и sample FPS;
  - prompt и Remember prompt;
  - progress/timer/cancel/error;
  - summary output;
  - frame grid;
  - Save summary as comment.

## 7. Monitoring / CLIP Probes

### 7.1 Probe board

- [x] Есть список probes, channel filter и text search.
- [x] Есть create/edit/delete.
- [x] Есть API-действия start/stop analytics capture.
- [ ] **P0** Восстановить непрерывный probe run loop. Сейчас React вызывает
  `/probes/run` один раз после start capture, тогда как legacy повторяет query
  до остановки.
- [x] Есть benchmark.
- [x] Есть выбранный probe inspector.
- [ ] **P0** Добавить подтверждение удаления probe.
- [ ] **P0** Проверить enabled/disabled/running/paused/idle на реальном runtime.
- [ ] **P0** Не считать один analytics stream состоянием всех probes channel без
  явной модели поведения.
- [ ] **P0** Обновлять probe count/status в общей оболочке после CRUD и runtime
  действий.
- [ ] **P1** Вернуть просмотр recent hits, frames indexed и window.
- [ ] **P1** Добавить paging/ограничение списка hits.
- [ ] **P1** Показывать bookmark gate reason, remaining cooldown и dedupe state.
- [ ] **P1** Проверить benchmark error/result details и повторный запуск.

### 7.2 Probe settings

- [x] Есть name, channel, enabled.
- [x] Есть positive/negative prompt pairs.
- [x] Есть pos floor и margin.
- [x] Есть bookmarks, severity, cooldown и dedupe.
- [x] Есть image probe.
- [x] Есть ROI drawing.
- [x] Есть cast на несколько channels.
- [ ] **P0** Проверить полное round-trip сохранение всех полей backend Probe.
- [ ] **P0** Не терять `window_sec`, `top_k`, `fps` и другие поддерживаемые поля,
  даже если они скрыты в advanced section.
- [ ] **P0** Проверить координаты ROI при `object-fit`, letterbox и разных aspect
  ratios.
- [ ] **P0** Отображать реальный stream/capture/buffer status.
- [x] Probe status вычисляет duration из пары `time_range_ms=[first,last]`.
- [ ] **P0** Вернуть Start Stream в probe editor; сейчас доступен только Stop.
- [ ] **P0** Добавить безопасное поведение при смене channel с несохранённым ROI.
- [ ] **P1** Перенести cast option `Start streams`, если он остаётся
  поддерживаемым.
- [ ] **P1** Проверить conflict modes `skip/create/update` и частичные ошибки.
- [ ] **P1** Добавить guard от потери несохранённых изменений.
- [x] Probe save/cast без `bookmarks:create` удаляет только bookmark-поля,
  сохраняет severity и не отображает bookmark controls.
- [x] Существующий bookmarked probe без `bookmarks:create` явно read-only;
  Cast не предлагает `update`, чтобы не получить скрытый backend 403.
- [ ] **P2** Если Probe Snapshot поддерживается, перенести:
  - capture snapshot;
  - actual resolution;
  - export;
  - Set as image probe.

## 8. EVA Agent

### 8.1 Чат и сессии

- [x] Есть streaming chat.
- [x] Есть создание, открытие и удаление sessions.
- [x] Есть image attachment вместе с текстовым сообщением.
- [x] Image-only Agent message отправляется с системным текстом
  `Describe this image.`.
- [x] Есть model selector.
- [x] Есть список, создание, редактирование и запуск skills.
- [x] Есть action-plan approval/apply.
- [x] Есть Operator Mode и базовое управление Archive UI.
- [x] Добавлена видимая кнопка входа/выхода из full-screen.
- [ ] **P0** Либо вернуть отдельный Agent section, либо официально утвердить
  dock-only UX и удалить недостижимое состояние/код.
- [x] Добавлена Stop-кнопка, отменяющая текущий fetch через AbortController.
- [ ] **P0** Обрабатывать все terminal SSE events и гарантированно снимать busy.
- [x] Incremental SSE parser обрабатывает partial frames, LF/CRLF, malformed
  event и остаток buffer после EOF.
- [x] Persisted assistant/tool rows и trusted action receipts восстанавливаются
  в Research trace при повторном открытии session.
- [ ] **P0** Проверить CSRF, session expiry и reconnect во время stream.
- [ ] **P1** Показывать tool/context budget events.
- [ ] **P1** Добавить понятный статус stalled/heartbeat/timeout.
- [ ] **P1** Добавить подтверждение удаления session.

### 8.2 Tool result cards и управление консолью

- [ ] **P0** Сопоставить все tool names старого UI с React renderer.
- [ ] **P0** Вернуть специализированные карточки для:
  - Archive search/detections;
  - detection summary;
  - channels и channel survey;
  - probes/create/update/delete/deploy;
  - prompt settings;
  - describe frame;
  - video summaries и counts;
  - visual state transitions;
  - bookmark;
  - report.
- [ ] **P0** Показывать approval preview и итоговый receipt без потери полей.
- [ ] **P0** Проверить rendering evidence thumbnails, IDs, scores и coverage.
- [ ] **P0** Санитизировать markdown и значения tool result.
- [x] Agent → UI mirroring расширен на Archive, Video и Probes через закрытые
  server-derived `ui_effects`; модель не генерирует DOM/UI-команды.
- [x] Подтверждённые probe/prompt mutations обновляют или открывают
  соответствующий React workspace без полного reload; preview не выдаётся за
  применённое состояние.
- [x] Current console scope передаётся отдельно как валидируемый
  `console_context`, а не дописывается prose-префиксом в пользовательскую
  реплику. Явный запрос оператора имеет приоритет над UI defaults.
- [ ] **P1** Показывать operator context, который фактически отправляется Agent,
  в диагностическом режиме.

## 9. Settings

### 9.1 Общие настройки

- [x] Есть Server, Search, Models, Advanced и Luxriot sections.
- [x] Есть поля archive retention и форма estimate inputs.
- [x] Archive capacity отображает `estimate.daily/retained/bytes` и
  `current.row_count` по реальной backend schema.
- [x] Есть Environment editor.
- [x] Визуальная иерархия всех вкладок Settings разгружена: большие вложенные
  карточки секций, жёсткие разделители header/sidebar/footer и плиточная подсветка
  навигации заменены единым рабочим полотном, локальными световыми маркерами и
  мягкими фейдами. Контуры сохранены только у интерактивных полей, действий и
  настоящих табличных данных; встроенный Appearance использует тот же каркас.
- [ ] **P0** Сверить все writable keys старой формы с React и backend `/settings`.
- [ ] **P0** Добавить отсутствующие поддерживаемые model options, включая
  подтверждённые SigLIP варианты.
- [ ] **P0** Проверить `indexFolderName/indexMode` и другие legacy settings:
  перенести либо официально удалить.
- [ ] **P0** Реализовать условную видимость/disabled-state зависимых полей:
  embedder, fusion, DINO, rerank, segments и experimental flag.
- [ ] **P0** Не отправлять скрытые/неподдерживаемые значения только потому, что
  они остались в local state.
- [ ] **P0** Сохранить write-only поведение Luxriot password.
- [ ] **P0** Добавить dirty-state guard при закрытии Settings.
- [ ] **P0** Валидировать диапазоны и взаимосвязанные значения до POST.
- [ ] **P1** Пересчитывать archive capacity по изменённым estimate inputs, а не
  только загружать один первоначальный результат.
- [ ] **P1** Показывать какие изменения требуют restart.
- [ ] **P1** Проверить Reset to defaults: preview до Save и отсутствие
  случайного немедленного применения.

### 9.2 Users и Sessions

- [x] Есть список, создание и редактирование users.
- [x] Есть roles, active state, password и allowed channels; камеры отображаются
  отдельными карточками с названием и ID, а checkbox-состояния имеют единый
  checked/hover/disabled/focus UI.
- [x] Есть действие revoke всех sessions пользователя.
- [x] Revoke response нормализует `revokedSessions` в отображаемый count.
- [x] Users editor использует channel picker с All/None/Refresh; снятие одного
  channel при wildcard `*` корректно выбирает все доступные кроме него.
- [x] Явный channel selection проверяется по доступному списку; неизвестные или
  недоступные IDs не сохраняются.
- [ ] **P0** Вернуть отдельные действия Reset password и Enable/Disable с
  понятным подтверждением.
- [x] Текущий account нельзя отключить из Users editor.
- [x] Для выбранного пользователя загружается полный session inventory.
- [x] Добавлены Active only filters для users и sessions.
- [x] Добавлен revoke конкретной session.
- [x] Login и `/auth/me` возвращают `sessionId`; текущая session помечается и
  защищена от индивидуального revoke, а bulk revoke текущего account отключён.
- [x] Session inventory показывает last seen, expiry/revocation, client IP и
  user-agent из разрешённого backend response.
- [ ] **P1** Обновлять список после каждого действия без stale selection.

### 9.3 Audit

- [x] Audit response нормализует
  `occurredAt/actorUserId/targetType/targetId/channelId`, а Vite proxy включает
  `/audit`.
- [x] Audit permission hint использует фактический `audit:view`.
- [x] Audit реализует cursor/Next Page.
- [x] Следующая audit page добавляется к уже загруженным events.
- [x] Audit показывает loading/error/empty/denied как отдельные состояния.
- [x] Выбранное событие показывает полные metadata/details в JSON view.
- [ ] **P1** Добавить удобное копирование request ID и correlation данных.
- [ ] **P1** Проверить channel-scope и all-channel требования.

### 9.4 Environment

- [x] Есть load/save `.env`.
- [ ] **P0** Проверить masked secrets: `***` не должно затирать реальный secret.
- [ ] **P0** Предупреждать о restart и несохранённых изменениях.
- [ ] **P0** Не показывать Environment пользователю без разрешения.
- [ ] **P1** Показывать число сохранённых переменных и backend validation error.

## 10. API client и контракты

- [ ] **P0** Вынести все frontend API-вызовы в typed API modules.
- [x] Прямые `fetch` отсутствуют в компонентах; исключение — обоснованный
  streaming transport внутри `api/agent.ts`.
- [ ] **P0** Стандартизировать JSON/multipart/DELETE, CSRF и error parsing.
- [ ] **P0** Добавить единый тип ошибки: status, code, message, request ID.
- [ ] **P0** Обрабатывать non-JSON ошибки и пустые ответы.
- [ ] **P0** Добавить runtime guards/normalizers для критичных ответов, не
  полагаться только на TypeScript interfaces.
- [ ] **P0** Проверить все timestamps: seconds против milliseconds.
- [ ] **P0** Проверить все image references: base64, thumbnail endpoint,
  `/detections/image`, Windows/Linux server paths.
- [ ] **P0** Не использовать server filesystem path как browser URL.
- [ ] **P0** Добавить AbortController для search, preview и длинных запросов.
- [ ] **P1** Добавить retry только для безопасных read operations.
- [ ] **P1** Логировать frontend API failures без секретов и image payloads.

## 11. Security

- [ ] **P0** Проверить XSS для Agent markdown, VLM summaries, probe names,
  comments, audit details и backend error messages.
- [ ] **P0** Проверить CSRF всех mutation endpoints.
- [ ] **P0** Проверить отсутствие токенов/паролей в localStorage, logs и
  rendered DOM.
- [x] Permission-based rendering использует frontend helpers, а обязательная
  backend authorization повторно проверена `test_http_auth_routes` и
  `test_security_smoke`.
- [ ] **P0** Проверить channel isolation для Archive, Video, Monitoring, Agent и
  admin selectors.
- [ ] **P0** Добавить безопасные лимиты/валидацию image upload.
- [ ] **P0** Проверить object URL/base64 cleanup и ограничения памяти.
- [ ] **P1** Настроить CSP для production React assets.
- [ ] **P1** Проверить `target="_blank"`/external links, если они появятся.
- [ ] **P1** Провести security smoke под administrator, operator и ограниченным
  пользователем.

## 12. Автоматические тесты

Состояние на снимке аудита:

- [x] `npm run build`: TypeScript + Vite production build прошёл; bundle
  `362.42 kB` JS / `98.92 kB` CSS до gzip; локальные fixed-font assets
  загружаются отдельными fingerprinted файлами.
- [x] `npm test`: 40 React unit/contract tests прошли, включая appearance,
  structured console context, closed UI effects и восстановление tool traces.
- [x] Targeted Postgres identity/auth/security suite: 78 tests, `OK`
  (`3 skipped`) после добавления `sessionId` в login/me contract.
- [x] 17 существующих UI CSS contract checks прошли.
- [x] 85 из 85 запущенных API/auth/audit/security smoke tests вне Agent loop
  прошли.
- [x] Повторно запущены 23 archive/API dataflow smoke tests — все прошли.
- [x] Targeted Agent loop/auth/context/UI-effect suite: 93 tests и 13 subtests
  прошли; остаётся только локальное предупреждение CUDA 804 на несовместимом
  driver/runtime стенде.
- [x] В `react-ui/package.json` есть `vitest` и воспроизводимый `npm test`.
- [ ] **P1** Запланировать совместимое обновление Vite: полный `npm audit`
  показывает 1 moderate и 1 high у dev toolchain; production dependencies
  (`npm audit --omit=dev`) уязвимостей не показывают.
- [x] `pytest` доступен в текущем `.venv`; pytest-style suites запускаются
  напрямую через `.venv/bin/python -m pytest`.

### 12.1 Unit

- [ ] **P0** Detection normalizer: list/search schemas и отсутствующие поля.
- [x] Archive filter serialization для list/text/image и custom/preset time.
- [ ] **P0** Timestamp/score/severity formatting.
- [x] Permission и channel-scope helpers покрыты unit tests.
- [x] SSE parser с chunk boundaries, CRLF, malformed event и EOF.
- [ ] **P0** Probe form round-trip и ROI normalization.
- [ ] **P0** Settings serialization, password masking и defaults.

### 12.2 Contract/integration

- [ ] **P0** Auth login/me/logout/expired/revoked.
- [x] Archive list/text/image используют общий filter serializer.
- [x] Archive pagination serializes `limit/offset`; backend dataflow smoke
  подтверждает route contract.
- [ ] **P0** Describe/Find similar/full image.
- [ ] **P0** Video start/stop/flush/session/rollups/runtime.
- [ ] **P0** Probe CRUD/run/cast/status.
- [ ] **P0** Agent streaming/session/approval.
- [ ] **P0** Settings/users/sessions/audit/environment.
- [ ] **P0** 401/403/404/409/429/500 и non-JSON error responses.

### 12.3 End-to-end

- [ ] **P0** Добавить browser E2E suite для production build, не только Vite dev.
- [ ] **P0** Проверить основные сценарии administrator.
- [ ] **P0** Проверить operator без admin permissions.
- [ ] **P0** Проверить пользователя с ограниченным списком channels.
- [ ] **P0** Проверить пустую систему без channels/probes/detections.
- [ ] **P0** Проверить Luxriot offline/degraded.
- [ ] **P0** Проверить reload во время активной/истёкшей session.
- [ ] **P1** Добавить screenshot/visual regression для ключевых экранов.
- [ ] **P1** Добавить accessibility smoke: keyboard, focus, labels, contrast.

## 13. Производительность и устойчивость

- [ ] **P0** Измерить initial JS/CSS size и время первого отображения на целевом
  operator workstation.
- [ ] **P0** Проверить память при длительной работе Video feed и Agent chat.
- [ ] **P0** Ограничить количество одновременно декодированных base64 thumbnails.
- [ ] **P0** Проверить polling: streams, feed, probe status и previews не должны
  дублироваться после remount.
- [ ] **P0** Останавливать timers, polling и in-flight requests при смене экрана.
- [ ] **P0** Проверить 50-channel deployment и длительную сессию.
- [ ] **P1** Добавить code splitting для тяжёлых разделов/модальных окон.
- [ ] **P1** Виртуализировать длинные Audit, Agent и feed lists при необходимости.
- [ ] **P1** Проверить slow network и backend latency без блокировки всего UI.

## 14. Документация и эксплуатация

- [ ] **P0** Обновить deployment guide: Node version, `npm ci`, build, asset path.
- [ ] **P0** Обновить offline USB/patch scripts и runbooks.
- [ ] **P0** Добавить React smoke в preflight/predeploy acceptance.
- [ ] **P0** Обновить operator guide/screenshots.
- [ ] **P0** Описать browser support и целевые разрешения.
- [ ] **P0** Описать feature flag переключения старого/нового UI и rollback.
- [ ] **P1** Добавить troubleshooting для blank screen, stale assets, CSRF,
  401/403 и Vite/prod различий.
- [ ] **P1** Обновить release notes и known limitations.

## 15. Cutover и удаление старого UI

- [ ] **P0** Развернуть React за feature flag на тестовом стенде.
- [ ] **P0** Пройти весь acceptance checklist на production-like данных.
- [ ] **P0** Выполнить пилот с реальными operator/admin ролями.
- [ ] **P0** Сравнить frontend/API error rate старого и нового UI.
- [ ] **P0** Подтвердить, что rollback проверен до переключения по умолчанию.
- [ ] **P0** Переключить default entrypoint на React.
- [ ] **P0** Оставить legacy fallback только на заранее определённый период.
- [ ] **P0** После периода стабилизации удалить:
  - `templates/index.html`;
  - runtime route `/js/app.js`;
  - `static/js/app.js`;
  - неиспользуемые legacy CSS selectors/assets;
  - Flask template variables, нужные только старому UI.
- [ ] **P0** Перед удалением повторно проверить, что legacy JS не содержит
  единственную реализацию поддерживаемого сценария.
- [ ] **P0** Удалить feature flag fallback и обновить runbooks.
- [ ] **P0** Зафиксировать итоговый parity/retirement report.

## 16. Финальные acceptance-сценарии

- [ ] Administrator входит, видит разрешённые разделы и корректный system status.
- [ ] Operator с ограниченными channels нигде не видит чужие channels или данные.
- [ ] Archive list работает с channel/source/probe/time и динамической
  подгрузкой при скролле без пропусков/дубликатов.
- [ ] Text и image search применяют тот же набор фильтров.
- [ ] Archive Inspector открывает full frame, Describe и Find similar.
- [ ] Video-summary result открывается как batch review и переходит в VLM feed.
- [ ] Live summaries стартуют, flush выполняется, feed и runtime обновляются.
- [ ] Rollup drill-down/back, live-follow, bookmark, copy/export работают.
- [ ] Stream stop/pause actions корректно отражаются в Video и Monitoring.
- [ ] Probe создаётся, настраивается, запускается, даёт hit и удаляется.
- [ ] ROI и image probe сохраняются без изменения координат/данных.
- [ ] Cast даёт корректный результат при create/update/skip и partial failure.
- [ ] Agent стримит ответ, показывает tool cards и применяет approved plan.
- [ ] Agent session и skill переживают reload.
- [ ] Settings сохраняются, secrets не раскрываются и restart requirement виден.
- [ ] User создаётся, ограничивается channels, отключается и получает revoke.
- [ ] Audit filters и pagination возвращают ожидаемые события.
- [ ] Истёкшая/revoked session возвращает пользователя на Login.
- [ ] Luxriot offline и backend errors отображаются без blank screen.
- [ ] Production build открывается после чистой установки и после patch upgrade.
- [ ] Rollback на предыдущий UI/build проверен и документирован.

## Связанные файлы

- Старый интерфейс: `templates/index.html`, `static/js/app.js`,
  `static/css/app.css`.
- React-прототип: `react-ui/src`.
- Backend entrypoint и routes: `oldapp.py`.
- Archive API map: `docs/frontend_rewrite/archive_api_map.md`.
- Production configuration: `docs/00_CANON/config_reference.md`.
- Deployment guide: `docs/install/deployment_guide.md`.
