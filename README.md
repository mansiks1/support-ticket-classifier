# Support Ticket Classifier

Исследование классификации англоязычных банковских обращений: качество,
вычислительная стоимость и отказ от автоматического решения при неуверенности.

Статус: опубликован проверенный каркас; эксперименты ещё не выполнены.
Неподтверждённых метрик нет. Источник: [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets),
данные CC BY 4.0, собственный код MIT. Приоритет не предсказывается: меток нет.

## Установка

Python 3.12:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
```

Протокол до экспериментов: [PROTOCOL.md](PROTOCOL.md).
План и фактическая история: [reports/progress.md](reports/progress.md).
