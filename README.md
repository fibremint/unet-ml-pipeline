
# Data directory structure
```
.
├── preprocess-checkpoint.json  - (auto generated)
├── preprocess-metadata         - (auto generated)
│   ├── 2_00223_sub0.svs.json
│   ├── 2_00225_sub0.svs.json
│   └── ...
├── region-annotation
│   ├── negative
│   │   ├── 1_00061_sub0
│   │   ├── 1_00062_sub0
│   │   └── ...
│   └── positive
│       ├── 1_00061_sub0
│       ├── 1_00062_sub0
│       └── ...
├── slide
│   ├── test
│   │   └── 2_00232_sub0.svs
│   └── train
│       ├── 2_00223_sub0.svs
│       ├── 2_00225_sub0.svs
│       ├── 2_00225_sub1.svs
│       ├── 2_00245_sub0.svs
│       └── 2_00248_sub0.svs
└── slide-patch                - (auto generated)
    └── train
        ├── ground-truth
        └── image

```

# Reference
https://github.com/zizhaozhang/nmi-wsi-diagnosis