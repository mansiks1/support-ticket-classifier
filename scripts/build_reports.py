"""Render prose, tables and plots from persisted measured evidence."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / "reports" / name).read_text(encoding="utf-8"))


def write(name, content):
    path = ROOT / name
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(content.strip() + "\n", encoding="utf-8", newline="\n")


def table(headers, rows):
    return (
        "| "
        + " | ".join(headers)
        + " |\n| "
        + " | ".join(["---"] * len(headers))
        + " |\n"
        + "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows)
    )


rows = read("experiments.json")
transformer = read("transformer.json")
all_rows = rows + [transformer]
selection = read("selection.json")
test = read("test-results.json")
cal = read("calibration.json")
audit = read("data-audit.json")
splits = read("splits.json")
environment = read("environment.json")
policy = test["practical"]["selective"]
score = test["practical"]["official"]
clean_score = test["practical"]["decontaminated"]
svm = max(
    [r for r in rows if r["config"]["method"] == "svm"], key=lambda r: r["validation"]["macro_f1"]
)
base = next(r for r in rows if r["name"] == "lr_base")
comparison = table(
    [
        "Метод",
        "Validation macro-F1",
        "Δ к LR",
        "Обучение, с",
        "Один запрос, мс",
        "Batch, мс/текст",
        "Размер, МБ",
        "Поддержка",
        "Вывод",
    ],
    [
        [
            r["name"],
            f"{r['validation']['macro_f1']:.4f}",
            f"{r['validation']['macro_f1'] - base['validation']['macro_f1']:+.4f}",
            f"{r['train_seconds']:.3f}",
            f"{r['single_ms']:.3f}",
            f"{r['batch_ms_per_item']:.3f}",
            f"{r['model_bytes'] / 1e6:.3f}",
            "torch + tokenizer + weights"
            if r["config"]["method"] == "transformer"
            else ("sklearn + WordNet" if r["name"] == "lr_lemma" else "sklearn + vocabulary"),
            "Выбран по протоколу"
            if r["name"] == selection["practical_model"]
            else (
                "Компактная альтернатива"
                if r["name"] == svm["name"]
                else ("Нулевая точка" if r["name"] == "dummy" else "Не выбран")
            ),
        ]
        for r in all_rows
    ],
)
summary = table(
    ["Выборка", "Accuracy", "Macro-F1", "Weighted-F1", "Top-2", "Macro AP"],
    [
        [name]
        + [
            f"{record[k]:.4f}"
            for k in ["accuracy", "macro_f1", "weighted_f1", "top2_accuracy", "macro_ap"]
        ]
        for name, record in [
            ("Validation (до калибровки)", transformer["validation"]),
            ("Official test", score),
            ("Test без близких совпадений", clean_score),
        ]
    ],
)
cal_table = table(
    ["Состояние", "NLL", "Brier (сумма по классам)", "ECE (10 bins)"],
    [
        [key] + [f"{cal[key][m]:.4f}" for m in ["nll", "brier", "ece"]]
        for key in ["before", "after"]
    ],
)
perclass = read("test-per-class.json")
worst = sorted(
    [(label, value) for label, value in perclass.items() if label in selection["classes"]],
    key=lambda item: item[1]["f1-score"],
)[:10]
worst_table = table(
    ["Класс", "Precision", "Recall", "F1", "Support"],
    [
        [
            label,
            f"{v['precision']:.3f}",
            f"{v['recall']:.3f}",
            f"{v['f1-score']:.3f}",
            int(v["support"]),
        ]
        for label, v in worst
    ],
)
thresholds = table(
    ["Порог", "Принято", "Coverage", "Accuracy принятых", "Оператору"],
    [
        [
            r["threshold"],
            r["accepted"],
            f"{r['coverage']:.3f}",
            f"{r['accuracy']:.3f}" if r["accuracy"] is not None else "—",
            r["human"],
        ]
        for r in cal["thresholds"]
        if r["threshold"] in [0, 0.5, 0.7, 0.75, 0.8, 0.9, 0.95, 0.99]
    ],
)
cv = f"""- Разработал воспроизводимый классификатор банковских обращений по 77 категориям: сравнил 17 классических конфигураций и DistilBERT, получил test macro-F1 {score["macro_f1"]:.4f}.
- Реализовал калибровку и отказ от автоматического решения: на отложенном test автоматически принято {policy["coverage"]:.2%} обращений с точностью {policy["accuracy"]:.2%}; порог выбран только по validation.
- Разделил обучение, калибровку, validation и test с контролем групп дубликатов; отдельно оценил результат после исключения {test["near_or_exact_overlap_rows"]} близких совпадений с development.
- Упаковал модель в FastAPI, добавил Docker, 41 автоматический тест и GitHub Actions; подготовил технический отчёт, анализ ошибок и воспроизводимые notebook.
- Измерил компромисс качества и стоимости: выигрыш DistilBERT над SVM составил {transformer["validation"]["macro_f1"] - svm["validation"]["macro_f1"]:.4f} macro-F1 на validation при росте размера модели в {transformer["model_bytes"] / svm["model_bytes"]:.1f} раза.
"""
value = """Проект демонстрирует постановку прикладной ML-задачи; работу с реальными несовершенными данными и дисбалансом; построение baseline; честное разделение train/calibration/validation/test и предотвращение data leakage; сравнение нескольких семейств моделей и выбор метрик; анализ ошибок и ограничений; оценку уверенности; сравнение качества с вычислительной и инженерной стоимостью; воспроизводимость экспериментов; вынесение логики notebook в программные модули; автоматические тесты, API, контейнеризацию, техническое письмо и объяснение результатов нетехническому читателю. Числа подтверждены сохранёнными запусками; опыт эксплуатации сервиса под реальной нагрузкой не заявляется."""
write(
    "reports/resume.md",
    "# Формулировки для резюме\n\n" + cv + "\n## Почему проект подходит для резюме\n\n" + value,
)
write(
    "reports/technical-report.md",
    f"""# Классификация обращений: качество и цена усложнения

## Аннотация

На BANKING77 выполнены {len(rows)} классических конфигураций и один двухэпоховый
DistilBERT. Исследовательская и практическая модели совпали: `{selection["practical_model"]}`.
Official test macro-F1 — **{score["macro_f1"]:.4f}**, после исключения близких
совпадений с development — **{clean_score["macro_f1"]:.4f}**. Это benchmark для
англоязычных банковских запросов, а не доказательство готовности к автономной работе.

## Постановка задачи

Пользователь системы — команда поддержки, маршрутизирующая обращения. Вход —
один английский текст или пакет до 64 текстов. Выход — категория (null при отказе),
предполагаемая категория, confidence, requires_human и priority=null. Меток
приоритета в данных нет. Неверная маршрутизация может задержать разбор потери
карты или спорного платежа. Потенциальная польза — сокращение ручной сортировки;
экономия времени операторов в реальном сервисе не измерялась.

## Данные, лицензия и очистка

Источник: [PolyAI](https://github.com/PolyAI-LDN/task-specific-datasets), BANKING77,
Casanueva et al., *Efficient Intent Detection with Dual Sentence Encoders*, 2020,
[публикация](https://arxiv.org/abs/2003.04807). Данные CC BY 4.0; собственный код MIT;
базовые веса DistilBERT Apache-2.0. Зафиксированы upstream revision и SHA-256.
Поля — text и category, {len(audit["class_counts"])} классов. Приватные данные не использованы.

Из {audit["raw_rows"]} development-строк удалено {audit["conflicting_rows_removed"]}
строки с конфликтующими нормализованными текстами и {audit["normalised_duplicates_removed"]}
последующих повторов. Осталось {audit["clean_rows"]}. Невалидных строк и null-ячеек:
{audit["invalid_rows"]} и {audit["null_cells"]}. Точное сравнение исходных текстов
не находило повторов: они проявились после NFKC, casefold и нормализации пунктуации.
Проверка email и длинных цифровых последовательностей дала ноль срабатываний;
это ограниченный экран, не полная гарантия отсутствия чувствительной информации.
Публикуются только небольшие вручную просмотренные примеры с атрибуцией.

Число объектов в классе: от {min(audit["class_counts"].values())} до {max(audit["class_counts"].values())}.
Все классы и количества: [data-audit.json](data-audit.json). Медиана длины —
{audit["text_length_quantiles"]["0.5"]:.0f} символов, 95-й процентиль —
{audit["text_length_quantiles"]["0.95"]:.1f}, максимум — {audit["text_length_quantiles"]["1.0"]:.0f}.

![EDA](figures/eda.png)

## Протокол и защита от утечки

Протокол опубликован до обучения: [PROTOCOL.md](../PROTOCOL.md). Seed=2026.
Компоненты близких дубликатов построены по char_wb 3–5 cosine ≥0.92:
{splits["nontrivial_components"]} нетривиальных компонент, {splits["near_duplicate_pairs"]}
пар. StratifiedGroupKFold удерживает компоненты внутри частей. Получено 5992 fit,
1996 calibration и 1983 validation, во всех частях есть 77 классов. Пересечения
групп и нормализованных текстов проверены. Предиктивный TF-IDF обучался только
на fit; представление для поиска дубликатов не использовалось как признак модели.
При отсутствии user/dialogue/time ID проверить временную и пользовательскую
утечку нельзя. В признаки не попадают category, индекс строки и служебные поля.

Validation использован для ограниченного поиска и инженерного выбора; calibration
— только для температуры. Выбор, веса и порог хешированы и опубликованы до
загрузки test. После открытия test обучение и порог не менялись. Повторная команда
`evaluate --verify` проверяет воспроизводимость замороженных результатов, а не
выбирает решение. Все split hashes: [splits.json](splits.json).

## Эксперименты и метрики

Dummy предсказывает частый класс. Далее TF-IDF + Logistic Regression, Linear SVM,
ComplementNB и DistilBERT. Основная метрика macro-F1 одинаково учитывает каждый
класс. Accuracy может скрывать слабые редкие классы; weighted-F1 дополняет, но
не заменяет macro-F1. Top-2 полезен как список подсказок оператору. Macro AP —
one-vs-rest average precision, не трапецеидальная площадь под PR-кривой; ROC-AUC
не нужен как дублирующая метрика. Precision/recall и confusion сохранены для каждого класса.

Поиск: SVM C ∈ {{0.5,1,2}}, LR C ∈ {{0.5,1,2}}, NB alpha ∈ {{0.1,1}}.
Восемь однофакторных LR-абляций: регистр, пунктуация, стоп-слова, WordNet noun
lemmatisation, word unigrams, char_wb 3–5, словарь 5000 и balanced weights.
Все сравнения используют одинаковые части. Взаимодействия факторов не исследованы.
Встроенный word-tokenizer уже отбрасывает пунктуацию — потому явное удаление
пунктуации здесь не дало выигрыша. WordNet используется без POS tagging.

{comparison}

Classical timings: CPU, один поток, измерение fit без загрузки пакетов и данных.
Latency включает преобразование текста и предсказание; медиана пяти прогретых
измерений. Размер — сжатый joblib для классики и каталог сохранённых весов с
токенизатором для Transformer, поэтому форматы различаются. Dummy также содержит
TF-IDF для единого интерфейса; его стоимость не минимально возможная.
RSS — память всего процесса, не изолированный пик; поля есть в experiments.json.
Обучение Transformer: RTX 4050, две эпохи, AdamW 5e-5, batch 16, max_length 96;
GPU peak allocated {transformer["peak_gpu_mb"]:.1f} MiB. GPU и CPU latency нельзя
трактовать как сравнение на одном устройстве. Версии: [environment.json](environment.json),
PyTorch {transformer["torch"]}; CPU {environment["processor"]}, RAM {environment["ram_gb"]:.1f} GiB.
Скачивание 3.2 GiB CUDA runtime и весов не включено во время обучения.
Дополнительный CPU serving benchmark: медиана одиночного запроса 28.86 мс,
batch=32 — 37.29 мс на текст; cold load и первый запрос — 62.32 с, RSS 806.97 MiB.
Это отдельный прогретый замер, а не нагрузочный тест; см. cpu-serving.json.
JSON/CSV с параметрами, версиями и хешами служат лёгким журналом экспериментов;
MLflow не добавлялся ради малого числа запусков.

![Validation comparison](figures/comparison.png)

## Какие усложнения оправдались

LR существенно превосходит Dummy и задаёт рабочую отправную точку. SVM улучшает
LR на {svm["validation"]["macro_f1"] - base["validation"]["macro_f1"]:.4f} macro-F1,
обучается быстрее в этом эксперименте и хранится компактнее. Для поддержки оба
требуют лишь sklearn и словарь; линейные коэффициенты доступны для анализа.
ComplementNB дешевле учится, но уступает по качеству, поэтому не выбран.
Лемматизация увеличила время и зависимости ради умеренного выигрыша; stop words
ухудшили качество. Символьная модель помогает относительно базового LR, но
уступает SVM и крупнее. Балансировка дала умеренный прирост, а ограничение словаря
сократило размер с небольшой потерей качества. Простые unigrams оказались лучше
базовых bigrams: усложнение признаков не автоматически полезно.

DistilBERT дал +{transformer["validation"]["macro_f1"] - svm["validation"]["macro_f1"]:.4f}
macro-F1 к лучшему SVM, но весит в {transformer["model_bytes"] / svm["model_bytes"]:.1f}
раза больше; его fit занял в {transformer["train_seconds"] / svm["train_seconds"]:.1f}
раза больше времени даже с GPU. Поддержка сложнее: torch, transformers, tokenizer,
веса и ограничения длины. Коэффициенты заменены приближённым анализом чувствительности.
Правило практического выбора: минимальный размер среди моделей в пределах 0.01
от лучшего macro-F1. В этот допуск SVM не попал, поэтому оба победителя — DistilBERT.
Это явное ограничение на допустимую потерю качества, а не универсальный бизнес-оптимум.
Если сервис допускает потерю 0.0162 macro-F1 на validation ради компактности,
SVM остаётся разумным альтернативным решением. Эксперимент не доказывает, что
Transformer всегда лучше или всегда неоправдан.

## Калибровка и передача оператору

SVM decision_function не является вероятностью. Raw softmax служил только
нормированным score для сравнения ранжирования. Для выбранного DistilBERT
температура {selection["temperature"]:.6f} обучена минимизацией calibration NLL;
она сохранена, потому что validation NLL улучшился. Top-1 решения не меняются.

{cal_table}

{thresholds}

![Calibration and coverage](figures/calibration.png)

Порог {selection["policy"]["threshold"]:.2f} выбран как максимальное покрытие при
validation accepted accuracy ≥0.95 и хотя бы 100 принятых примерах. Это точность
среди принятых, не per-class precision и не SLA. На test принято {policy["accepted"]}
из {test["test_rows"]} ({policy["coverage"]:.2%}), оператору передано {policy["human"]};
accepted accuracy {policy["accuracy"]:.2%}, Wilson 95% интервал
[{policy["accuracy_wilson_95"][0]:.2%}; {policy["accuracy_wilson_95"][1]:.2%}].
Интервал предполагает независимость наблюдений; близкие формулировки ограничивают её.

## Единственная финальная test-оценка

{summary}

Из {test["test_rows"]} официальных примеров {test["exact_overlap_rows"]} совпали
с development после нормализации; всего {test["near_or_exact_overlap_rows"]} имели
точное/близкое совпадение. Заранее предусмотренный sensitivity-анализ оставил
{test["decontaminated_rows"]} примеров. Их macro-F1 ниже. Это важная угроза
валидности исходного benchmark; порог 0.92 не обнаруживает все парафразы, а удаление
меняет распределение выборки. На test не сравнивались все 18 конфигураций:
оценён только замороженный финальный выбор. Полный поиск сравнивается по validation.

Хуже всего по test F1:

{worst_table}

Полные precision/recall: [test-per-class.json](test-per-class.json),
[confusion](test-confusion.csv). Слабые классы приведены описательно, без изменения модели.

## Ошибки, интерпретация и устойчивость

[Отдельный разбор](error-analysis.md): 10 правильных, 20 ошибочных, 10 неуверенных
примеров; смешанные намерения, недостаток контекста и гипотезы о разметке.
[Топ-признаки SVM](top-features.json) для всех классов; [удаление слов](occlusion.json)
для Transformer — только локальная чувствительность, не причинное объяснение.

![Validation confusion](figures/confusion.png)

Регистр и лишние пробелы устойчивы на проверенном validation; искусственная
опечатка снизила macro-F1 до 0.8013. Пустой/слишком длинный ввод отклоняется.
Модель не умеет надёжно обнаруживать OOD: вопрос о погоде принят с confidence
0.8913. Русский и смешанный ввод не поддерживаются, несмотря на отсутствие падения.
Тексты ограничены 4000 символами, но модель видит лишь первые 96 токенов.

## Реализация, проверки и угрозы валидности

Исследование отделено от src и api. FastAPI: /health, /predict, /model-info,
строгая проверка типов, размер пакета 1–64. Отказ возвращает category=null и
сохраняет suggested_category как подсказку. Тексты не сохраняются в логах.
41 тест включает unit, train smoke, roundtrip, reproducibility и полный путь
обучение → сохранение → загрузка → HTTP через TestClient. CI также собирает
Docker и проверяет HTTP в контейнере на явно синтетическом fixture.
Фактические проверки итоговых весов и контейнера записываются в [verification.json](verification.json).

Ограничения: один seed, один ограниченный neural run, выбранные гиперпараметры,
selection bias от нескольких validation-сравнений, отсутствие внешнего банка,
неизвестных тем, временных/user ID и приоритетов. Статистическая значимость малых
различий не установлена. Нет нагрузочного теста, аутентификации, rate limiting
или проверки реальных бизнес-потерь; это воспроизводимый портфельный сервис.

Дальше: отдельный OOD-набор и правила эскалации, несколько seeds, группировка
семантических парафразов, real-world temporal holdout, multilingual/multilabel,
CPU-оптимизация и измерение стоимости ошибок совместно с поддержкой.

## Воспроизведение

Команды установки, загрузки release-артефактов, полного исследования в отдельной
копии, проверки frozen metrics, API и Docker: [README](../README.md).
Числа и графики этого отчёта генерируются `python scripts/build_reports.py` из
сохранённых результатов. Ручная интерпретация ошибок вынесена в отдельный файл.
""",
)
write(
    "article/article.md",
    f"""# Стоят ли 1,6 пункта F1 нейросети: эксперимент с обращениями в поддержку

Представьте очередь банковской поддержки. Одно сообщение сообщает о задержке
карты, другое — о чужом платеже. Автоматическая сортировка может помочь, но
ошибочное уверенное решение способно отправить человека не к тому специалисту.
Поэтому я исследовал не только правильность категории, но и право модели отказаться.

## Сначала договоримся о честном эксперименте

Я взял открытый английский BANKING77: 77 намерений, от активации карты до
проблем с переводом. Источник — PolyAI и Casanueva et al. (2020), лицензия CC BY 4.0.
Это учебный benchmark, не выгрузка моего банка. Русские обращения и приоритеты
этот проект не моделирует: соответствующих меток нет.

Даже здесь данные не оказались идеально чистыми. Точное сравнение не нашло
повторов, а нормализация нашла 30 дубликатов и две конфликтующие строки.
После очистки я оставил отдельные части для обучения, калибровки и выбора модели,
удержав близкие дубликаты вместе. Официальный test оставался закрыт до выбора.
Так оценка не превращается в соревнование по запоминанию отложенной выборки.

## Простое решение уже полезно

Dummy, всегда выбирающий частую метку, дал macro-F1 0.0005. TF-IDF и Logistic
Regression дали {base["validation"]["macro_f1"]:.4f}. Но первое улучшение пришло
не от нейросети: Linear SVM достиг {svm["validation"]["macro_f1"]:.4f}, обучаясь
за {svm["train_seconds"]:.2f} секунды на одном CPU-потоке.

Я проверил восемь привычных способов предобработки по одному. Удаление стоп-слов
ухудшило результат. Удаление пунктуации ничего не изменило: стандартный tokenizer
уже её отбрасывал. Даже добавление word bigrams оказалось не гарантированным
улучшением — unigrams выиграли у исходной конфигурации. Такие результаты полезнее
списка «обязательных» шагов очистки: каждый шаг должен оправдать себя измерением.

## Нейросеть всё же выиграла — вопрос в цене

DistilBERT обучался две эпохи на RTX 4050. Validation macro-F1 стал
{transformer["validation"]["macro_f1"]:.4f}: +{100 * (transformer["validation"]["macro_f1"] - svm["validation"]["macro_f1"]):.2f}
процентного пункта к SVM. Обучение заняло {transformer["train_seconds"]:.1f} секунды,
а артефакт вырос примерно с {svm["model_bytes"] / 1e6:.2f} до {transformer["model_bytes"] / 1e6:.1f} МБ.
Кроме весов появились torch, tokenizer и ограничения длины. GPU latency и CPU
latency измерены на разных устройствах; выдавать их за чистое аппаратно равное
сравнение было бы нечестно.

Моё заранее записанное правило разрешало потерять максимум 0.01 macro-F1 ради
меньшего размера. SVM не уложился в этот допуск, поэтому итоговым стал DistilBERT.
Если продукт допускает потерю около 0.016, выбор может быть другим. Нельзя сначала
увидеть результат, а потом подобрать критерий так, чтобы победил любимый алгоритм.

![Измеренное сравнение](../reports/figures/comparison.png)

## Уверенность — тоже гипотеза

На отдельной calibration-части я подобрал температуру вероятностей. Она не меняет
победивший класс, но меняет силу уверенности. ECE на validation уменьшилась с
{cal["before"]["ece"]:.4f} до {cal["after"]["ece"]:.4f}. Затем выбрал порог 0.75,
при котором точность принятых validation-решений достигала хотя бы 95%.

Когда все решения были зафиксированы, test дал macro-F1 **{score["macro_f1"]:.4f}**.
Система приняла **{policy["coverage"]:.2%}** запросов с точностью **{policy["accuracy"]:.2%}**,
остальные {policy["human"]} из 3080 передала человеку. Это измеренная точность
внутри benchmark, не обещание качества на любом входе.

Проверка пересечений обнаружила ещё одну неприятную деталь: у {test["near_or_exact_overlap_rows"]}
test-текстов есть близкие или точные совпадения с development. На оставшихся
{test["decontaminated_rows"]} примерах macro-F1 снизился до {clean_score["macro_f1"]:.4f}.
Поэтому рядом с красивым официальным числом необходимо публиковать и этот результат.

## Где сервис ошибается

«Where is my card?» не объясняет, была ли карта заказана или потеряна.
«Why can't I make a transfer?» не говорит причину отказа, но модель уверенно
предположила проблему с получателем. Сообщение о взломанном аккаунте и чужом платеже
действительно подходит нескольким категориям, хотя benchmark требует одну.

Самый показательный провал — вопрос о погоде. Модель приняла его за вопрос о
сроках доставки карты с confidence 0.8913. Значит, temperature scaling и порог
не заменяют обнаружение неизвестных тем. Это ограничение я сохранил в отчёте,
вместо того чтобы добавить правило специально под один уже увиденный пример.

## Из notebook в API

Исследовательская логика находится в Python-модулях. Notebook лишь показывает
проверенные данные, таблицы и выводы. FastAPI принимает один текст или список;
при отказе `category` равна null, а `suggested_category` остаётся подсказкой
оператору. Поле `priority` также null: модель не должна изобретать отсутствующие метки.

```bash
curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' \\
  -d '{{"text":"My card has not arrived yet"}}'
```

Тесты проверяют не только метрики, но и путь от обучения маленькой модели до
HTTP-ответа, загрузку артефактов, ошибочные типы, пустые строки и Unicode.
Для развёртывания есть Docker, а для повторения — версии зависимостей, исходных
данных, seed и полный журнал экспериментов.

## Практический вывод

В этом эксперименте усложнение до SVM дало особенно выгодный шаг. DistilBERT
добавил качество, но потребовал заметно больше памяти и инфраструктуры. Часть
обычной предобработки оказалась бесполезной или вредной. Ценность проекта —
в измеренном выборе с открытыми ограничениями, а не в обязательной победе
самого простого или самого сложного подхода.

[Код и инструкции](https://github.com/mansiks1/support-ticket-classifier) ·
[Полный технический отчёт](../reports/technical-report.md) ·
[Примеры ошибок](../reports/error-analysis.md).
""",
)
write(
    "MODEL_CARD.md",
    f"""# Model card

- Final model: {selection["practical_model"]}, fine-tuned distilbert-base-uncased (Apache-2.0).
- Task: English banking intent classification, 77 classes; single label.
- Dataset: BANKING77, PolyAI / Casanueva et al. 2020, CC BY 4.0; no ownership claim.
- Training: 5992 fit samples, seed 2026, 2 epochs, AdamW 5e-5, batch 16, 96 tokens.
- Calibration: 1996 samples; temperature {selection["temperature"]:.6f}; validation threshold 0.75.
- Official test macro-F1: {score["macro_f1"]:.4f}; decontaminated test: {clean_score["macro_f1"]:.4f}.
- Test accepted accuracy: {policy["accuracy"]:.2%} at {policy["coverage"]:.2%} coverage.
- Intended use: supervised portfolio demonstration and in-domain routing experiments.
- Not intended: autonomous financial decisions, fraud detection, priority assessment,
  unrestricted multilingual or open-domain customer support.
- Known failure: weather question accepted as card delivery; confidence is not an OOD guarantee.
- Input limit: 4000 characters / 96 model tokens; batch max 64. Priority unavailable.
- No raw request text is persisted. API has no authentication or rate limiting.
- Load joblib only from trusted release after verifying SHA-256. Weight hashes:
  [selection.json](reports/selection.json). Details: [technical report](reports/technical-report.md).
""",
)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
main = [
    next(r for r in rows if r["name"] == "dummy"),
    base,
    svm,
    next(r for r in rows if r["name"] == "nb_a1"),
    transformer,
]
labels = [r["name"] for r in main]
axes[0].barh(
    labels,
    [r["validation"]["macro_f1"] for r in main],
    color=["#b8c4ce", "#267a92", "#267a92", "#267a92", "#df8350"],
)
axes[0].set(xlabel="Validation macro-F1", xlim=(0, 1), title="Measured quality")
axes[1].barh(labels, [r["model_bytes"] / 1e6 for r in main], color="#267a92")
axes[1].set(
    xlabel="Saved artifact MB (log scale)", xscale="log", title="Storage cost; different formats"
)
fig.tight_layout()
fig.savefig(ROOT / "reports/figures/comparison.png", dpi=130)
plt.close(fig)
print("Generated technical report, article, model card, resume and comparison plot")
