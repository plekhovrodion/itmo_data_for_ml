# Отчёт ActiveLearningAgent

Стратегия: `entropy`, label_col: `label_final`

Кривая обучения: `/Users/rodion/Desktop/itmo_data_for_ml-1/data/processed/learning_curve.png`

## История (accuracy / F1 macro vs n_labeled)

```json
[
  {
    "iteration": 0,
    "n_labeled": 50,
    "accuracy": 0.2,
    "f1": 0.04004554004554005
  },
  {
    "iteration": 1,
    "n_labeled": 70,
    "accuracy": 0.24,
    "f1": 0.05423497815368582
  },
  {
    "iteration": 2,
    "n_labeled": 90,
    "accuracy": 0.28,
    "f1": 0.08745920745920746
  },
  {
    "iteration": 3,
    "n_labeled": 110,
    "accuracy": 0.33,
    "f1": 0.1295530771957571
  },
  {
    "iteration": 4,
    "n_labeled": 130,
    "accuracy": 0.3,
    "f1": 0.12800874339335877
  },
  {
    "iteration": 5,
    "n_labeled": 150,
    "accuracy": 0.41,
    "f1": 0.2376545684810763
  }
]
```

**Примечание:** в `run_cycle` метки в пуле известны симулятору (оракул); в проде здесь был бы второй HITL.
