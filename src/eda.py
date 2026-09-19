"""Regenerate development-only distribution figures."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import PROCESSED, REPORTS


def main():
    frame = pd.read_csv(PROCESSED / "development.csv")
    folder = REPORTS / "figures"
    folder.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    frame.category.value_counts().sort_values().plot.bar(ax=axes[0], width=1)
    axes[0].set(
        xticks=[],
        xlabel="77 intents (sorted by count)",
        ylabel="Examples",
        title="Development class balance",
    )
    axes[1].hist(frame.text.str.len(), bins=45, color="#176b87")
    axes[1].set(xlabel="Characters", ylabel="Examples", title="Development text lengths")
    fig.tight_layout()
    fig.savefig(folder / "eda.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
