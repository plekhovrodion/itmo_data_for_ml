#!/usr/bin/env python3
"""Одна команда: Kaggle (₽/м²) + etagi (100 строк) → data/raw/*.csv"""

from agents.data_collection_agent import DataCollectionAgent


def main() -> None:
    agent = DataCollectionAgent("config.yaml")
    paths = agent.run()
    print("Готово:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
