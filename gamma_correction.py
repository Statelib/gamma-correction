"""
=============================================================================
  ГАММА-КОРРЕКЦИЯ ИЗОБРАЖЕНИЯ — реализация без сторонних библиотек
=============================================================================

Что такое гамма-коррекция?
──────────────────────────
Человеческий глаз воспринимает яркость нелинейно: мы гораздо лучше различаем
оттенки в тёмных областях, чем в светлых. Мониторы и камеры тоже работают
нелинейно. Гамма-коррекция позволяет компенсировать эти отклонения.

Математика:
    output = (input / 255)^(1/γ) * 255

    γ < 1  → изображение светлеет (поднимаем тени)
    γ = 1  → без изменений
    γ > 1  → изображение темнеет (убираем пересветы)

LUT (Look-Up Table — таблица соответствий):
    Вместо вычисления формулы для каждого из миллионов пикселей,
    мы один раз заранее считаем результат для всех 256 возможных
    значений (0–255) и сохраняем в массив (LUT).
    Потом для любого пикселя просто делаем lut[pixel_value] —
    это O(1) вместо O(n) вычислений.

Поддерживаемые форматы: BMP, PPM/PGM (бинарный P5/P6)
=============================================================================
"""

import struct
import math
import os
import sys


# ─────────────────────────────────────────────────────────────────────────────
#  1. ПОСТРОЕНИЕ ТАБЛИЦЫ LUT
# ─────────────────────────────────────────────────────────────────────────────

def build_gamma_lut(gamma: float) -> list[int]:
    """
    Строит таблицу замены (LUT) для гамма-коррекции.

    Алгоритм для каждого значения i в диапазоне [0, 255]:
        1. Нормализуем: normalized = i / 255.0          → [0.0 … 1.0]
        2. Применяем степень: corrected = normalized ^ (1/gamma)
        3. Денормализуем: output = round(corrected * 255)
        4. Клэмп: ограничиваем [0, 255] на случай float-погрешностей

    Параметры:
        gamma — коэффициент гамма (float > 0)

    Возвращает:
        Список из 256 целых чисел — заменяющие значения яркости.
    """
    if gamma <= 0:
        raise ValueError(f"Гамма должна быть > 0, получено: {gamma}")

    inv_gamma = 1.0 / gamma
    lut = []

    for i in range(256):
        # Нормализация в [0, 1]
        normalized = i / 255.0

        # Применение степенной функции
        # math.pow обрабатывает 0.0 корректно (возвращает 0.0)
        corrected = math.pow(normalized, inv_gamma)

        # Денормализация и округление до целого
        value = int(round(corrected * 255.0))

        # Защита от выхода за пределы [0, 255]
        value = max(0, min(255, value))
        lut.append(value)

    return lut


# ─────────────────────────────────────────────────────────────────────────────
#  2. ЧТЕНИЕ / ЗАПИСЬ PPM (Portable PixMap — простой текстовый/бинарный формат)
# ─────────────────────────────────────────────────────────────────────────────

def read_ppm(path: str) -> tuple[int, int, list[list[list[int]]]]:
    """
    Читает бинарный PPM (P6) или PGM (P5) файл.

    Структура PPM/P6:
        Магическое число "P6\n"
        Ширина Высота\n
        Максимальное значение (обычно 255)\n
        Бинарные данные: 3 байта на пиксель (R, G, B)

    Возвращает:
        (width, height, pixels)
        pixels[y][x] = [R, G, B] — список пикселей
    """
    with open(path, "rb") as f:
        raw = f.read()

    # Парсим заголовок PPM (ASCII-часть до бинарных данных)
    header = []
    i = 0
    while len(header) < 4:
        # Пропускаем комментарии (строки начинающиеся с #)
        if raw[i:i+1] == b'#':
            while raw[i:i+1] not in (b'\n', b''):
                i += 1
        elif raw[i:i+1] in (b' ', b'\t', b'\n', b'\r'):
            i += 1
        else:
            # Читаем токен (число или магическое слово)
            token = b''
            while raw[i:i+1] not in (b' ', b'\t', b'\n', b'\r', b''):
                token += raw[i:i+1]
                i += 1
            if token:
                header.append(token.decode('ascii'))

    magic, width_s, height_s, maxval_s = header[0], header[1], header[2], header[3]
    width, height = int(width_s), int(height_s)
    maxval = int(maxval_s)

    # i сейчас указывает на первый байт после последнего токена заголовка,
    # но нужно пропустить ровно один разделитель перед данными
    i += 0  # уже после пробела/новой строки

    channels = 3 if magic == 'P6' else 1
    expected = width * height * channels

    pixel_data = raw[i:i + expected]
    if len(pixel_data) < expected:
        raise ValueError(f"Файл повреждён: ожидалось {expected} байт данных, получено {len(pixel_data)}")

    # Строим двумерный массив пикселей
    pixels = []
    idx = 0
    for y in range(height):
        row = []
        for x in range(width):
            if magic == 'P6':
                r = pixel_data[idx]
                g = pixel_data[idx + 1]
                b = pixel_data[idx + 2]
                idx += 3
                row.append([r, g, b])
            else:  # P5 — оттенки серого
                v = pixel_data[idx]
                idx += 1
                row.append([v, v, v])
        pixels.append(row)

    return width, height, pixels


def write_ppm(path: str, width: int, height: int, pixels: list[list[list[int]]]) -> None:
    """
    Записывает пиксели в бинарный PPM (P6) файл.

    Параметры:
        path    — путь для сохранения
        width   — ширина в пикселях
        height  — высота в пикселях
        pixels  — pixels[y][x] = [R, G, B]
    """
    with open(path, "wb") as f:
        # Заголовок (ASCII)
        header = f"P6\n{width} {height}\n255\n"
        f.write(header.encode('ascii'))

        # Бинарные данные пикселей — 3 байта на пиксель
        for row in pixels:
            for pixel in row:
                f.write(bytes(pixel))


# ─────────────────────────────────────────────────────────────────────────────
#  3. ЧТЕНИЕ / ЗАПИСЬ BMP (Windows Bitmap)
# ─────────────────────────────────────────────────────────────────────────────

def read_bmp(path: str) -> tuple[int, int, list[list[list[int]]]]:
    """
    Читает 24-битный BMP файл (наиболее распространённый формат без сжатия).

    Структура BMP:
        [0x00] 2 байта  — магия "BM"
        [0x02] 4 байта  — размер файла
        [0x06] 4 байта  — зарезервировано
        [0x0A] 4 байта  — смещение до пиксельных данных
        [0x0E] 4 байта  — размер заголовка (обычно 40)
        [0x12] 4 байта  — ширина
        [0x16] 4 байта  — высота (отрицательная = top-down)
        [0x1C] 2 байта  — битов на пиксель (24 для RGB)
        ...
        Пиксели хранятся снизу вверх (если высота > 0), с выравниванием строк
        до 4 байт (padding).
    """
    with open(path, "rb") as f:
        data = f.read()

    # Проверяем магические байты
    if data[0:2] != b'BM':
        raise ValueError("Не является BMP файлом (нет сигнатуры 'BM')")

    # Читаем заголовок через struct.unpack_from(format, buffer, offset)
    # '<' — little-endian (стандарт для BMP)
    # 'I' — unsigned int 32-bit, 'i' — signed int, 'H' — unsigned short 16-bit
    pixel_offset = struct.unpack_from('<I', data, 0x0A)[0]
    width        = struct.unpack_from('<i', data, 0x12)[0]
    height       = struct.unpack_from('<i', data, 0x16)[0]
    bit_count    = struct.unpack_from('<H', data, 0x1C)[0]

    if bit_count != 24:
        raise ValueError(f"Поддерживается только 24-битный BMP, обнаружен {bit_count}-битный")

    # Определяем порядок строк
    flip = height > 0   # BMP с положительной высотой хранит строки снизу вверх
    height = abs(height)

    # Выравнивание строки до кратного 4 байтам
    # Формула: (width * 3 + 3) // 4 * 4
    row_size = (width * 3 + 3) // 4 * 4
    padding = row_size - width * 3

    pixels = []
    for y in range(height):
        row_idx = (height - 1 - y) if flip else y
        offset = pixel_offset + row_idx * row_size
        row = []
        for x in range(width):
            # BMP хранит цвета в порядке BGR, а не RGB!
            b = data[offset + x * 3 + 0]
            g = data[offset + x * 3 + 1]
            r = data[offset + x * 3 + 2]
            row.append([r, g, b])
        pixels.append(row)

    return width, height, pixels


def write_bmp(path: str, width: int, height: int, pixels: list[list[list[int]]]) -> None:
    """
    Записывает пиксели в 24-битный BMP файл.

    Формируем заголовок вручную через struct.pack и записываем пиксели
    в порядке BGR снизу вверх (стандарт BMP).
    """
    row_size = (width * 3 + 3) // 4 * 4  # выравнивание
    padding = row_size - width * 3
    pixel_data_size = row_size * height
    file_size = 54 + pixel_data_size  # 54 = размер заголовков BMP

    buf = bytearray()

    # ── Файловый заголовок (14 байт) ──
    buf += b'BM'                              # 2: сигнатура
    buf += struct.pack('<I', file_size)        # 4: размер файла
    buf += struct.pack('<HH', 0, 0)           # 4: зарезервировано
    buf += struct.pack('<I', 54)              # 4: смещение до пикселей

    # ── DIB заголовок BITMAPINFOHEADER (40 байт) ──
    buf += struct.pack('<I', 40)              # 4: размер заголовка
    buf += struct.pack('<i', width)           # 4: ширина
    buf += struct.pack('<i', -height)         # 4: высота (отриц. = top-down)
    buf += struct.pack('<H', 1)              # 2: плоскости цвета
    buf += struct.pack('<H', 24)             # 2: битов на пиксель
    buf += struct.pack('<I', 0)              # 4: сжатие (0 = нет)
    buf += struct.pack('<I', pixel_data_size) # 4: размер данных пикселей
    buf += struct.pack('<i', 2835)           # 4: пикс/м по X (~72 DPI)
    buf += struct.pack('<i', 2835)           # 4: пикс/м по Y
    buf += struct.pack('<I', 0)              # 4: цветов в палитре
    buf += struct.pack('<I', 0)              # 4: важных цветов

    # ── Пиксельные данные (BGR, строки выровнены до 4 байт) ──
    for row in pixels:
        for r, g, b in row:
            buf += bytes([b, g, r])          # BMP хранит BGR!
        buf += bytes(padding)                # нулевой padding

    with open(path, "wb") as f:
        f.write(buf)


# ─────────────────────────────────────────────────────────────────────────────
#  4. ПРИМЕНЕНИЕ ГАММА-КОРРЕКЦИИ
# ─────────────────────────────────────────────────────────────────────────────

def apply_gamma_correction(
    pixels: list[list[list[int]]],
    lut: list[int]
) -> list[list[list[int]]]:
    """
    Применяет гамма-коррекцию к каждому пикселю через LUT.

    Для каждого пикселя [R, G, B]:
        new_R = lut[R]
        new_G = lut[G]
        new_B = lut[B]

    Мы применяем одну и ту же LUT ко всем каналам — это корректно
    для большинства изображений. Для точной работы с цветовым
    пространством sRGB нужна отдельная линеаризация, но для
    практических задач этого достаточно.

    Время: O(width × height) — одна операция индексирования на канал.
    Память: создаём новый массив пикселей, оригинал не изменяется.
    """
    result = []
    for row in pixels:
        new_row = []
        for r, g, b in row:
            new_row.append([lut[r], lut[g], lut[b]])
        result.append(new_row)
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  5. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ─────────────────────────────────────────────────────────────────────────────

def detect_format(path: str) -> str:
    """Определяет формат файла по расширению."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.bmp':
        return 'bmp'
    elif ext in ('.ppm', '.pgm'):
        return 'ppm'
    else:
        raise ValueError(f"Неподдерживаемый формат: '{ext}'. Поддерживаются: .bmp, .ppm, .pgm")


def load_image(path: str):
    """Загружает изображение, автоматически определяя формат."""
    fmt = detect_format(path)
    if fmt == 'bmp':
        return read_bmp(path)
    else:
        return read_ppm(path)


def save_image(path: str, width: int, height: int, pixels) -> None:
    """Сохраняет изображение, автоматически определяя формат по расширению."""
    fmt = detect_format(path)
    if fmt == 'bmp':
        write_bmp(path, width, height, pixels)
    else:
        write_ppm(path, width, height, pixels)


def print_lut_preview(lut: list[int], gamma: float, steps: int = 8) -> None:
    """Выводит в консоль сводку таблицы LUT для наглядности."""
    print(f"\n  LUT-preview (γ = {gamma}):")
    print(f"  {'Вход':>6}  {'Выход':>6}  {'Изменение':>10}")
    print(f"  {'─'*6}  {'─'*6}  {'─'*10}")
    indices = [int(i * 255 / (steps - 1)) for i in range(steps)]
    for i in indices:
        diff = lut[i] - i
        sign = '+' if diff > 0 else ''
        print(f"  {i:>6}  {lut[i]:>6}  {sign}{diff:>9}")
    print()


def compute_statistics(pixels) -> dict:
    """
    Считает среднюю яркость и гистограмму изображения.
    Используется для отчёта до/после коррекции.
    """
    total = 0
    count = 0
    histogram = [0] * 256

    for row in pixels:
        for r, g, b in row:
            # Яркость по формуле ITU-R BT.601
            brightness = int(0.299 * r + 0.587 * g + 0.114 * b)
            total += brightness
            count += 1
            histogram[brightness] += 1

    avg = total / count if count > 0 else 0
    return {'avg_brightness': avg, 'pixel_count': count, 'histogram': histogram}


# ─────────────────────────────────────────────────────────────────────────────
#  6. ТЕСТ-ГЕНЕРАТОР: создаём тестовое изображение без внешних зависимостей
# ─────────────────────────────────────────────────────────────────────────────

def create_test_image(path: str, width: int = 256, height: int = 128) -> None:
    """
    Создаёт тестовое BMP/PPM с градиентом для демонстрации гамма-коррекции.

    Изображение содержит:
        - Горизонтальный градиент яркости (слева = чёрный, справа = белый)
        - Цветные полосы: серая / красная / зелёная / синяя
    """
    pixels = []
    section = height // 4

    for y in range(height):
        row = []
        for x in range(width):
            v = int(x / (width - 1) * 255)  # значение градиента 0..255

            if y < section:             # серый градиент
                row.append([v, v, v])
            elif y < section * 2:       # красный
                row.append([v, 0, 0])
            elif y < section * 3:       # зелёный
                row.append([0, v, 0])
            else:                       # синий
                row.append([0, 0, v])

        pixels.append(row)

    save_image(path, width, height, pixels)
    print(f"  Тестовое изображение создано: {path} ({width}×{height})")


# ─────────────────────────────────────────────────────────────────────────────
#  7. ОСНОВНАЯ ФУНКЦИЯ
# ─────────────────────────────────────────────────────────────────────────────

def gamma_correction(input_path: str, output_path: str, gamma: float) -> None:
    """
    Полный конвейер гамма-коррекции:
        1. Загрузить изображение
        2. Построить LUT
        3. Применить LUT к каждому пикселю
        4. Сохранить результат
        5. Вывести отчёт

    Параметры:
        input_path  — путь к исходному изображению (.bmp или .ppm)
        output_path — путь для сохранения результата
        gamma       — коэффициент гамма (float > 0)
    """
    print(f"\n{'═'*60}")
    print(f"  ГАММА-КОРРЕКЦИЯ ИЗОБРАЖЕНИЯ")
    print(f"{'═'*60}")
    print(f"  Вход:  {input_path}")
    print(f"  Выход: {output_path}")
    print(f"  γ    : {gamma}")

    # ── Шаг 1: загрузка ──────────────────────────────────────────
    print(f"\n  [1/4] Загрузка изображения...")
    width, height, pixels = load_image(input_path)
    print(f"        Размер: {width}×{height} пикселей ({width * height:,} px)")

    # ── Шаг 2: статистика ДО ─────────────────────────────────────
    stats_before = compute_statistics(pixels)
    print(f"        Средняя яркость до: {stats_before['avg_brightness']:.1f}")

    # ── Шаг 3: построение LUT ────────────────────────────────────
    print(f"\n  [2/4] Построение таблицы LUT (256 значений)...")
    lut = build_gamma_lut(gamma)
    print_lut_preview(lut, gamma)

    # ── Шаг 4: применение коррекции ──────────────────────────────
    print(f"  [3/4] Применение гамма-коррекции...")
    corrected = apply_gamma_correction(pixels, lut)

    # ── Шаг 5: сохранение ────────────────────────────────────────
    print(f"\n  [4/4] Сохранение результата...")
    save_image(output_path, width, height, corrected)

    # ── Отчёт ─────────────────────────────────────────────────────
    stats_after = compute_statistics(corrected)
    print(f"\n{'─'*60}")
    print(f"  РЕЗУЛЬТАТ:")
    print(f"  Средняя яркость:  {stats_before['avg_brightness']:.1f}  →  {stats_after['avg_brightness']:.1f}")
    delta = stats_after['avg_brightness'] - stats_before['avg_brightness']
    direction = "светлее" if delta > 0 else "темнее"
    print(f"  Изменение:        {delta:+.1f} ({direction})")
    size = os.path.getsize(output_path)
    print(f"  Файл сохранён:    {output_path} ({size:,} байт)")
    print(f"{'═'*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  ТОЧКА ВХОДА
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    Использование:
        python gamma_correction.py <input> <output> <gamma>
        python gamma_correction.py --demo               # создаёт и обрабатывает тестовое изображение

    Примеры:
        python gamma_correction.py photo.bmp bright.bmp 0.5   # осветлить
        python gamma_correction.py photo.bmp dark.bmp   2.0   # затемнить
        python gamma_correction.py photo.ppm out.ppm    1.8   # классическая Mac-гамма
        python gamma_correction.py --demo
    """
    args = sys.argv[1:]

    if not args or args[0] in ('-h', '--help'):
        print(main.__doc__)
        return

    if args[0] == '--demo':
        # Режим демонстрации — генерируем тестовое изображение и применяем несколько гамм
        print("\n  РЕЖИМ ДЕМОНСТРАЦИИ")
        create_test_image("test_original.ppm")

        for g in [0.4, 0.7, 1.0, 1.5, 2.2]:
            out = f"test_gamma_{str(g).replace('.', '_')}.ppm"
            gamma_correction("test_original.ppm", out, g)
        print("  Демо завершено! Откройте файлы test_gamma_*.ppm для сравнения.")
        return

    if len(args) != 3:
        print("Ошибка: нужно 3 аргумента: <input> <output> <gamma>")
        print("Для справки: python gamma_correction.py --help")
        sys.exit(1)

    input_path, output_path, gamma_str = args

    try:
        gamma = float(gamma_str)
    except ValueError:
        print(f"Ошибка: гамма должна быть числом, получено '{gamma_str}'")
        sys.exit(1)

    if not os.path.exists(input_path):
        print(f"Ошибка: файл не найден: '{input_path}'")
        sys.exit(1)

    gamma_correction(input_path, output_path, gamma)


if __name__ == '__main__':
    main()
