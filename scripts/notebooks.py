"""Generate and execute compact notebooks; all training logic lives in src/."""

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
setup = """from pathlib import Path
import json
import pandas as pd
ROOT = Path.cwd()
if not (ROOT / 'reports').exists():
    ROOT = ROOT.parent
"""
notebooks = {
    "01_eda.ipynb": [
        (
            "markdown",
            "# BANKING77: аудит и разбиение\nДанные PolyAI / Casanueva et al. (2020), CC BY 4.0. Только development; test не читается. Подготовка: `python -m src.data prepare` и `python -m src.data split`.",
        ),
        ("code", setup),
        (
            "code",
            "frame = pd.read_csv(ROOT / 'data/processed/development.csv')\nframe[['text', 'category']].info()\nframe.text.str.len().describe()",
        ),
        (
            "code",
            "audit = json.loads((ROOT / 'reports/data-audit.json').read_text())\n{k: v for k,v in audit.items() if k not in ['class_counts', 'text_length_quantiles']}",
        ),
        (
            "code",
            "counts = frame.category.value_counts()\nprint('Class count:', len(counts), 'min:', counts.min(), 'max:', counts.max())\ncounts.to_frame('examples').head(15)",
        ),
        (
            "code",
            "parts = {n: pd.read_csv(ROOT / f'data/processed/{n}.csv') for n in ['train', 'calibration', 'validation']}\nfor a,b in [('train','validation'),('train','calibration'),('calibration','validation')]:\n    assert not set(parts[a].group) & set(parts[b].group)\npd.DataFrame({n: {'rows':len(p), 'classes':p.category.nunique()} for n,p in parts.items()}).T",
        ),
        (
            "markdown",
            "![Development distributions](../reports/figures/eda.png)\n\nБлизкие дубликаты определяются по cosine ≥0.92 и объединяются в компоненты связности. Это не гарантия отсутствия семантических парафразов. Пользовательских и временных идентификаторов нет. Подробности: [DATA_CARD](../DATA_CARD.md).",
        ),
    ],
    "02_experiments.ipynb": [
        (
            "markdown",
            "# Эксперименты и цена усложнения\nNotebook читает результаты реально выполненных запусков. Обучение и оценка реализованы в `src`, повторные команды — в README. Test здесь не используется для выбора модели.",
        ),
        ("code", setup),
        (
            "code",
            "experiments = pd.read_csv(ROOT / 'reports/experiments.csv')\nexperiments[['name','macro_f1','train_seconds','single_ms','model_bytes']].sort_values('macro_f1', ascending=False)",
        ),
        (
            "code",
            "transformer = json.loads((ROOT / 'reports/transformer.json').read_text())\n{k:v for k,v in transformer.items() if k not in ['history']}",
        ),
        (
            "code",
            "selection = json.loads((ROOT / 'reports/selection.json').read_text())\n{k:selection[k] for k in ['research_model','practical_model','temperature','policy']}",
        ),
        (
            "code",
            "cal = json.loads((ROOT / 'reports/calibration.json').read_text())\npd.DataFrame({k:{m:cal[k][m] for m in ['nll','brier','ece']} for k in ['before','after']}).T",
        ),
        (
            "markdown",
            "![Calibration](../reports/figures/calibration.png)\n\nСначала выбор по validation, затем заморозка артефакта и один финальный запуск оценки test. Малые отличия одного seed не доказывают превосходства метода.",
        ),
        (
            "code",
            "examples = pd.read_csv(ROOT / 'reports/prediction-examples.csv')\nexamples[examples['sample'] == 'error'].head(10)",
        ),
        (
            "code",
            "final = json.loads((ROOT / 'reports/test-results.json').read_text())\nfinal['practical']",
        ),
        (
            "markdown",
            "Ручной разбор: [error-analysis](../reports/error-analysis.md). Интерпретация, ограничения и выводы: [технический отчёт](../reports/technical-report.md).",
        ),
    ],
}
folder = root / "notebooks"
folder.mkdir(exist_ok=True)
for name, cells in notebooks.items():
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        nbf.v4.new_markdown_cell(text) if kind == "markdown" else nbf.v4.new_code_cell(text)
        for kind, text in cells
    ]
    notebook.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    NotebookClient(notebook, timeout=120, resources={"metadata": {"path": str(root)}}).execute()
    # Strip absolute runtime paths from execution metadata; keep actual compact outputs.
    for cell in notebook.cells:
        cell.metadata.pop("execution", None)
    nbf.write(notebook, folder / name)
    print(f"Executed {name}: {sum(c.cell_type == 'code' for c in notebook.cells)} code cells")
