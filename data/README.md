# Data

Datasets are not committed to this repository. They must be downloaded and stored locally or on Google Drive.

## Datasets

| Dataset | Source | Notes |
|---|---|---|
| Kvasir-SEG | https://datasets.simula.no/kvasir-seg/ | 1000 images with masks |
| CVC-ClinicDB | https://polyp.grand-challenge.org/CVCClinicDB/ | Research and education use only |

## Expected Structure

```
data/
└── TestDataset/
    ├── Kvasir/
    │   ├── images/
    │   └── masks/
    └── CVC-ClinicDB/
        ├── images/
        └── masks/
```

## Notes

- PraNet expects the structure `TestDataset/<name>/images` and `masks`, so organize the folders this way.
- To compare with the published results, use the same test split used by the PraNet authors. Using a different split makes the comparison invalid.
- Image file names must be identical in `images/` and `masks/`.
- Respect the CVC-ClinicDB license: research and education use only.
