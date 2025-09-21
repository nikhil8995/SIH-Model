# SIH-Model

/SIH
├── ml/
│   ├── __init__.py
│   ├── config.py
│   ├── feature_extractor.py
│   ├── dataset.py
│   ├── train_probe.py
│   ├── infer.py
│   └── artifacts/
│       ├── linear_probe.joblib
│       ├── text_features.npy
│       ├── prompt_texts.npy
│       └── runtime_config.json
├── archive/
│   ├── classes.txt
│   └── dataset/
│       └── 00 Test/
│           ├── train/
│           │   ├── images/
│           │   └── labels/
│           ├── val/
│           │   ├── images/
│           │   └── labels/
│           └── test/
│               ├── images/
│               └── labels/
├── app.py
├── fix_graffiti_labels.py
└── requirements.txt
