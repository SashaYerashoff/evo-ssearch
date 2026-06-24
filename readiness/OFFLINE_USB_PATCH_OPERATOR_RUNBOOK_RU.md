# EVA AI: установка offline-патча с USB-накопителя

Дата: 2026-06-22

Этот runbook рассчитан на оператора на клиентской машине EVA AI без доступа в
интернет. Цель: привезти patch bundle на флешке, сделать резервные копии,
установить код, проверить `/health` и `/ready`, а при проблеме откатиться на
предыдущее состояние.

Не вставляйте в терминал пояснительный текст. Вставляйте только команды из
блоков. Не копируйте и не отправляйте содержимое `/etc/eva-ai/eva-ai.env`: там
могут быть пароли и DSN.

## 0. Что должно быть на флешке

На инженерной машине заранее собирается архив вида:

```text
eva-ai-patch-YYYYMMDD-HHMMSS.tar.gz
eva-ai-patch-YYYYMMDD-HHMMSS.tar.gz.sha256
```

Внутри архива:

```text
manifest.txt
repo/
scripts/install_patch.sh
scripts/verify_patch.sh
scripts/rollback.sh
scripts/set_site_ips.sh
scripts/client_diagnostics.sh
repo/readiness/OFFLINE_USB_PATCH_OPERATOR_RUNBOOK_RU.md
repo/readiness/CLIENT_DIAGNOSTICS_RUNBOOK.md
```

Если архива нет, собрать его из рабочей копии репозитория можно так:

```bash
cd /home/sasha/Projects/evo-ssearch
scripts/build_patch_bundle.sh --output-dir /tmp/eva-ai-usb
```

Скопируйте созданные `.tar.gz` и `.sha256` на USB-накопитель.

## 1. Подготовка на клиентской EVA AI машине

Вставьте флешку. Найдите путь к ней:

```bash
lsblk -f
```

В примерах ниже путь флешки обозначен как `/media/$USER/EVA_USB`. Замените его
на реальный путь.

Создайте рабочую директорию и скопируйте архив локально:

```bash
mkdir -p ~/eva-ai-patch
cp /media/$USER/EVA_USB/eva-ai-patch-*.tar.gz* ~/eva-ai-patch/
cd ~/eva-ai-patch
```

Проверьте контрольную сумму, если рядом есть `.sha256`:

```bash
sha256sum -c eva-ai-patch-*.tar.gz.sha256
```

Ожидаемый результат: строка заканчивается на `OK`.

Распакуйте архив:

```bash
tar -xzf eva-ai-patch-*.tar.gz
cd eva-ai-patch-*
cat manifest.txt
```

## 2. Предпроверка перед установкой

Проверьте текущее состояние сервиса:

```bash
scripts/verify_patch.sh \
  --service eva-ai \
  --base-url http://127.0.0.1:5000
```

Если `/ready` уже красный до установки, зафиксируйте это в акте работ. Патч всё
равно можно ставить, но после установки важно отличать старую проблему от новой.

Посмотрите свободное место для backup:

```bash
df -h /var/backups /opt/eva-ai /etc/eva-ai
```

Если на `/var/backups` мало места, остановитесь и согласуйте с инженером другой
путь через `--backup-root`.

## 3. Установка патча

Запустите установку от root:

```bash
sudo scripts/install_patch.sh \
  --app-dir /opt/eva-ai/evo-ssearch \
  --env-file /etc/eva-ai/eva-ai.env \
  --service eva-ai \
  --base-url http://127.0.0.1:5000
```

Что делает скрипт:

- создаёт backup в `/var/backups/eva-ai/patch-YYYYMMDD-HHMMSS`;
- сохраняет env-файл, systemd unit/drop-ins и текущий код;
- делает `pg_dump`, если доступен `pg_dump` и найден DSN или локальная база;
- останавливает `eva-ai`;
- копирует код из bundle в `/opt/eva-ai/evo-ssearch`;
- запускает `eva-ai`;
- проверяет `/health` и `/ready`.

Нормальный результат:

```text
OK: ...
OK: health endpoint returned HTTP 200
OK: ready endpoint returned HTTP 200
```

Если скрипт напечатал `FAIL`, не продолжайте ручные правки. Сохраните весь
вывод команды и переходите к разделу rollback.

## 4. Проверка после установки

Повторите проверку:

```bash
scripts/verify_patch.sh \
  --service eva-ai \
  --base-url http://127.0.0.1:5000 \
  --timeout 60
```

Проверьте systemd-журнал без вывода секретов:

```bash
sudo journalctl -u eva-ai -n 120 --no-pager
```

Откройте UI с операторского рабочего места и проверьте:

- страница загружается;
- вход выполняется штатным пользователем;
- `/health` зелёный;
- `/ready` зелёный или показывает только заранее известные внешние зависимости;
- Luxriot live preview работает на тестовом канале;
- VLM/agent endpoints доступны, если они должны быть включены на этом стенде.

Для текущей архитектуры live video descriptions должны работать в одном
`gunicorn` worker на EVA AI host. Не увеличивайте
`EVOSSEARCH_GUNICORN_WORKERS` выше `1` без отдельного инженерного согласования:
capture loops пока живут в памяти процесса.

## 5. Если на объекте изменились IP-адреса

Узнайте актуальные IP:

```bash
hostname -I
ip -br addr
```

Примените новые адреса к env-файлу. Подставьте реальные значения:

```bash
sudo scripts/set_site_ips.sh \
  --env-file /etc/eva-ai/eva-ai.env \
  --service eva-ai \
  --luxriot-ip 192.168.1.10 \
  --luxriot-port 8080 \
  --inference-a-ip 192.168.1.20 \
  --inference-b-ip 192.168.1.21 \
  --agent-base-url http://127.0.0.1:1234/v1 \
  --restart
```

Скрипт не меняет логины и пароли. Если изменились Luxriot credentials, правьте
их вручную через:

```bash
sudo nano /etc/eva-ai/eva-ai.env
```

После смены IP повторите:

```bash
scripts/verify_patch.sh \
  --service eva-ai \
  --base-url http://127.0.0.1:5000
```

## 6. Rollback без восстановления базы

Обычный rollback возвращает код, env и systemd unit из последнего backup. Базу
данных он не трогает.

```bash
sudo scripts/rollback.sh \
  --app-dir /opt/eva-ai/evo-ssearch \
  --env-file /etc/eva-ai/eva-ai.env \
  --service eva-ai \
  --base-url http://127.0.0.1:5000
```

Если нужно откатиться не на последний backup, укажите директорию явно:

```bash
sudo scripts/rollback.sh \
  --backup-dir /var/backups/eva-ai/patch-YYYYMMDD-HHMMSS \
  --app-dir /opt/eva-ai/evo-ssearch \
  --env-file /etc/eva-ai/eva-ai.env \
  --service eva-ai \
  --base-url http://127.0.0.1:5000
```

После rollback выполните проверку из раздела 4.

## 7. Rollback базы данных только по согласованию

Восстановление PostgreSQL dump является разрушительной операцией: текущие
данные в базе могут быть заменены состоянием на момент backup. Делайте это
только после явного согласования с ответственным инженером.

Команда требует отдельного подтверждения:

```bash
sudo EVA_PATCH_CONFIRM_DB_RESTORE=yes scripts/rollback.sh \
  --restore-db \
  --backup-dir /var/backups/eva-ai/patch-YYYYMMDD-HHMMSS \
  --app-dir /opt/eva-ai/evo-ssearch \
  --env-file /etc/eva-ai/eva-ai.env \
  --service eva-ai \
  --base-url http://127.0.0.1:5000
```

## 8. Что отправить инженеру после работ

Отправьте только безопасные артефакты:

```bash
cat ~/eva-ai-patch/eva-ai-patch-*/manifest.txt
sudo ls -la /var/backups/eva-ai
scripts/verify_patch.sh --service eva-ai --base-url http://127.0.0.1:5000
```

Не отправляйте `/etc/eva-ai/eva-ai.env`, PostgreSQL dump и полные логи, если в
них могут быть секреты.
