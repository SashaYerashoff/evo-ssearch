# EVA AI offline update: 01 - подготовка флешки и Linux-шпаргалка

Цель: подготовить USB-накопитель на Windows, найти его на Linux-клиенте,
смонтировать при необходимости и уверенно выполнить базовые терминальные
команды.

Целевой релиз: `β 0.8.3`  
Schema head: `20260614_0006`  
Миграция БД для `β 0.8.2.1 -> β 0.8.3`: **нет**

Не копируйте и не отправляйте содержимое `/etc/eva-ai/eva-ai.env`: там могут
быть пароли, DSN и адреса закрытой сети.

## 1. Что должно быть на флешке

На флешке должны лежать два файла:

```text
eva-ai-patch-0.8.3-YYYYMMDD-*.tar.gz
eva-ai-patch-0.8.3-YYYYMMDD-*.tar.gz.sha256
```

Ссылку на актуальный bundle добавляет инженер проекта: `[FIELD: bundle link]`.

Если ссылка указывает на один `.tar.gz`, рядом должен быть `.sha256` с тем же
именем и суффиксом `.sha256`.

## 2. Подготовка флешки на Windows

1. Вставьте USB-накопитель.
2. Скопируйте на него `.tar.gz` и `.tar.gz.sha256`.
3. Не распаковывайте архив на Windows. Копируйте именно архив целиком.
4. Не переименовывайте файлы.
5. Безопасно извлеките флешку через Windows "Safely Remove Hardware".

Опциональная проверка SHA256 в PowerShell:

```powershell
cd E:\
Get-FileHash .\eva-ai-patch-0.8.3-*.tar.gz -Algorithm SHA256
type .\eva-ai-patch-0.8.3-*.tar.gz.sha256
```

Хэш из `Get-FileHash` должен совпасть с хэшем из `.sha256`. Если не совпал,
не используйте этот носитель.

Рекомендуемый формат флешки: `exFAT` или `NTFS`. Для Linux важнее, чтобы файл
копировался как обычный архив; права внутри архива сохраняет `tar.gz`, а не
файловая система флешки.

## 3. Найти флешку на Linux

Вставьте флешку и выполните:

```bash
lsblk -f
```

Ищите устройство с типом `part`, файловой системой `exfat`, `ntfs`, `vfat` или
`ext4`, и размером вашей флешки.

Пример:

```text
sdb
└─sdb1 exfat EVA_USB  64G  /media/luxriot/EVA_USB
```

В этом примере:

- диск: `/dev/sdb`;
- раздел флешки: `/dev/sdb1`;
- путь, куда она уже смонтирована: `/media/luxriot/EVA_USB`.

Важно: монтировать нужно раздел, например `/dev/sdb1`, а не весь диск
`/dev/sdb`.

## 4. Если флешка смонтировалась автоматически

Проверьте файлы:

```bash
ls -lh /media/$USER/*/
find /media/$USER -maxdepth 2 -name 'eva-ai-patch-0.8.3-*.tar.gz*' -ls
```

Если видите оба файла, переходите к копированию:

```bash
mkdir -p ~/eva-ai-patch
cp /media/$USER/EVA_USB/eva-ai-patch-0.8.3-*.tar.gz* ~/eva-ai-patch/
cd ~/eva-ai-patch
sha256sum -c eva-ai-patch-0.8.3-*.tar.gz.sha256
```

Ожидаемый результат:

```text
eva-ai-patch-...tar.gz: OK
```

## 5. Если флешка не смонтировалась автоматически

Найдите раздел:

```bash
lsblk -f
```

Создайте точку монтирования:

```bash
sudo mkdir -p /mnt/eva-usb
```

Смонтируйте раздел. Замените `/dev/sdX1` на реальный раздел из `lsblk -f`.

```bash
sudo mount /dev/sdX1 /mnt/eva-usb
ls -lh /mnt/eva-usb
```

Скопируйте архив локально:

```bash
mkdir -p ~/eva-ai-patch
cp /mnt/eva-usb/eva-ai-patch-0.8.3-*.tar.gz* ~/eva-ai-patch/
cd ~/eva-ai-patch
sha256sum -c eva-ai-patch-0.8.3-*.tar.gz.sha256
```

После работ размонтируйте флешку:

```bash
cd ~
sudo umount /mnt/eva-usb
```

Если `umount` говорит `target is busy`, значит какой-то терминал находится
внутри `/mnt/eva-usb`. Выполните `cd ~` во всех терминалах и повторите.

## 6. Распаковать bundle

После успешной проверки SHA256:

```bash
cd ~/eva-ai-patch
tar -xzf eva-ai-patch-0.8.3-*.tar.gz
cd eva-ai-patch-0.8.3-*
cat manifest.txt
```

В manifest должно быть:

```text
version=β 0.8.3
wheelhouse=included
```

Если `wheelhouse=not_included`, установка может всё ещё пройти за счёт
существующего `.venv`, но для полностью offline-сценария это риск. Перейдите к
preflight из документа `02`.

## 7. Терминальная шпаргалка

### Навигация

```bash
pwd                 # где я сейчас
ls                  # список файлов
ls -lh              # список с размерами
ls -la              # включая скрытые файлы
cd /path/to/dir     # перейти в директорию
cd ~                # домой
cd -                # назад в предыдущую директорию
```

### Копирование и просмотр

```bash
cp source target        # скопировать файл
cp -a source target     # скопировать с атрибутами
mkdir -p dir            # создать директорию
cat file                # вывести файл
sed -n '1,80p' file     # первые 80 строк
less file               # просмотр с прокруткой, выход: q
```

### Архивы и checksum

```bash
tar -tzf file.tar.gz | head     # посмотреть содержимое архива
tar -xzf file.tar.gz            # распаковать
sha256sum -c file.tar.gz.sha256 # проверить контрольную сумму
```

### Диски и место

```bash
lsblk -f            # диски, разделы, точки монтирования
df -h               # свободное место
du -sh path         # размер директории/файла
```

### Сервис EVA AI

```bash
systemctl status eva-ai --no-pager -l
sudo systemctl stop eva-ai
sudo systemctl start eva-ai
sudo systemctl restart eva-ai
journalctl -u eva-ai -n 120 --no-pager
```

### HTTP-проверки

```bash
curl -sS http://127.0.0.1:5000/health
curl -sS http://127.0.0.1:5000/ready | jq
```

Если сервис слушает HTTPS с self-signed сертификатом:

```bash
curl -k -sS https://127.0.0.1:5443/health
```

### Полезные клавиши

```text
Tab       автодополнение пути/команды
Up/Down   история команд
Ctrl+C    остановить текущую команду
Ctrl+L    очистить экран
q         выйти из less/journalctl
```

## 8. Что делать дальше

После копирования и распаковки bundle переходите к:

```text
readiness/OFFLINE_USB_02_PREFLIGHT_DECISION_RU.md
```

