# Dataset

This folder is intentionally kept light for GitHub. Use the setup script to download and organize the public Brain MRI dataset:

```bash
python -m dataset.download_dataset
```

The project expects four classes:

- `glioma`
- `meningioma`
- `notumor`
- `pituitary`

Large image files are ignored by Git.
