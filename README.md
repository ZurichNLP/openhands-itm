# OpenHands — Icelandic Sign Language Extension

This repository is a research adaptation of [**OpenHands**](https://github.com/AI4Bharat/OpenHands) by AI4Bharat, developed as part of a master's thesis at the University of Zurich (UZH). It extends the original library with support for the **ÍTM** dataset (Icelandic Sign Language).

The original library and all its documentation remain fully applicable:
**[ReadTheDocs: OpenHands](https://openhands.readthedocs.io)**

---

## Principal differences from the original repository

### 1. ÍTM dataset support

A new dataset class `ITMDataset` (`openhands/datasets/isolated/itm.py`) was added for an Icelandic Sign Language (ÍTM) dataset, with the language code `icl`. The dataset is provided in three variants to support ablation studies (*link to follow*):

| Variant | File | Classes |
|---------|------|----------------|
| Full | `itm_data.json` | 849 |
| Trimmed | `itm_data_trimmed.json` | 117 |
| Minimal | `itm_data_minimal.json` | 22 |

### 2. Example configs and experiment scripts

#### Example configs (`examples/configs/itm/`)

Three example Hydra configs are provided for ÍTM. They are included as reference only — data paths and checkpoint directories must be adjusted before use.

| Config | Model | Setup |
|--------|-------|-------|
| `basic_decoupled_gcn.yaml` | Decoupled GCN | Supervised training on ÍTM data alone |
| `multilingual_decoupled_gcn.yaml` | Decoupled GCN | Multilingual training with unified vocabulary; ÍTM added as a 7th language (`icl`) alongside datasets from 6 sign languages provided in matching pose format by the OpenHands authors (see the [documentation](https://openhands.ai4bharat.org/en/latest/instructions/datasets.html))
| `multilingual_original_vocab_st_gcn.yaml` | ST-GCN | Multilingual training with per-language vocabulary; ÍTM included alongside the same set of datasets |

#### Experiment scripts (`experiment_scripts/`)

Matching Python training/testing scripts and SLURM shell scripts are provided for each setup. These are also examples only and require path and SLURM configuration adjustments.

| Directory | Purpose |
|-----------|---------|
| `basic_itm/` | Supervised training and testing on ÍTM |
| `multilingual/` | Multilingual training with unified vocabulary |
| `multilingual_original/` | Multilingual training with per-language vocabulary |
| `pretrain_dpc_itm/` | Self-supervised DPC pretraining including ÍTM |

#### Additional scripts (`scripts/`)

- `visualize_pose.py` — visualizes pose estimates from pose .pkl files as overlay on an MP4 video

### 3. Multilingual inference (`openhands/apis/inference.py`)

The original `InferenceModel` compared predicted class indices directly against ground-truth labels from the test dataloader. This works for monolingual models, but fails silently for multilingual ones: `ConcatDataset` maps every gloss to a **global** index during training (built by sorting the full union of all datasets' glosses), while each sub-dataset's own `id_to_gloss` only covers its local vocabulary. Without reconstruction of the global mapping at test time, every comparison produces a mismatch and accuracy is 0%.

The updated `InferenceModel` handles this automatically when the `multilingual` flag is set in the test pipeline config:

- **Global mapping reconstruction** — `_build_global_mapping()` replicates the exact logic of `ConcatDataset.read_glosses()`: it applies `lang_code__gloss` prefixing (or `normalized_class_mappings` normalization when `unify_vocabulary: true`) and sorts the union of all glosses alphabetically. This produces the same `gloss→id` mapping the model was trained against. The train pipeline config is used as the source so that the full training vocabulary is captured, not just the classes present in the test split.
- **Sub-dataset reconstruction** — because `ConcatDataset` deletes `self.datasets` after `__init__`, the sub-datasets are re-instantiated from the config with `only_metadata=True` (skipping pose loading) and cached for the lifetime of the `InferenceModel`.
- **Ground-truth labels** — `ConcatDataset.read_original_dataset()` already writes global indices into each sample, so no remapping is needed for GT; only predicted indices are resolved via the global mapping.
- **Monolingual paths** — all three evaluation methods (`test_inference`, `compute_test_accuracy`, `compute_test_avg_class_accuracy`) are unchanged in behaviour when `multilingual` is not set.

#### Required config change for multilingual testing

Add `multilingual: true` under the test dataset block in your config:

```yaml
data:
    test_pipeline:
        dataset:
            _target_: openhands.datasets.isolated.ConcatDataset
            splits: "test"
            inference_mode: false
            multilingual: true        # <-- required for correct multilingual inference
            unify_vocabulary: true    # must match the value used during training
            datasets:
                ...
```

Without this flag, `InferenceModel` falls back to the original monolingual behaviour and multilingual evaluation will produce 0% accuracy.

---

### 4. Per-sample prediction logging in `compute_test_accuracy`

`compute_test_accuracy` currently prints every prediction to stdout as it runs:

```python
# openhands/apis/inference.py  ~line 246
label = self._resolve_label(pred_index)
filename = batch["files"][i]
print(f"{label}:\t{filename}\t{score}")
```

This produces one line per test sample. To suppress it and only see the per-dataset accuracy summary, comment out those three lines.

---

### 5. Subdirectory-aware inference-mode file enumeration (`openhands/datasets/isolated/base.py`)

The original `enumerate_data_files` assumed all pose `.pkl` files sit directly in `root_dir`. Several datasets in the multilingual setup store files in nested subdirectories, which caused `RuntimeError: No files found` at startup in inference mode. The method was updated to handle these structures:

| Dataset | Structure | Handling |
|---------|-----------|----------|
| GSL, MSASL | `root_dir/<gloss>/` | One level of subdirectories enumerated |
| INCLUDE | `root_dir/<category>/<gloss>/` | Two levels of subdirectories enumerated |
| All others | `root_dir/` (flat) | Unchanged |

Additional fixes applied in the same method:
- **Empty `root_dir` guard** — `ConcatDataset` passes `root_dir=""` and its file enumeration is handled separately; the method now returns immediately when `dir` is empty rather than crashing.
- **Per-directory `.pkl` fallback** — if a search directory contains no video files, any pre-existing `.pkl` files in that directory are collected directly. The fallback is now evaluated per directory and appends (`extend`) rather than replacing the accumulated list.

---

### 6. Optional validation pipeline (`openhands/core/data.py`)

The original `DataModule.setup()` unconditionally instantiated a `valid_dataset` from `valid_pipeline`, so configs without that block would crash at startup. `valid_pipeline` is now optional.

**Behaviour when `valid_pipeline` is omitted from the config:**

- `valid_dataset` is set to the same object as `train_dataset` — the full training set is used for validation callbacks (loss, accuracy) during training.
- `val_dataloader()` falls back to the `train_pipeline.dataloader` config for batch size and worker settings.
- Validation still runs every epoch as normal; it is not skipped.

This makes it straightforward to combine training and validation splits into a single training set (e.g. by pointing `train_pipeline` at a dataset file that merges both splits) without having to provide a dummy `valid_pipeline`.

**Behaviour when `valid_pipeline` is present** is unchanged: a separate validation dataset is instantiated and the usual channel/class-count assertions are enforced.

To use a combined train+val dataset, simply omit `valid_pipeline` from the config and point `train_pipeline` at the combined split:

```yaml
data:
    train_pipeline:
        dataset:
            splits: "train+val"   # or whichever combined split file you use
            ...
    # valid_pipeline:             # omit entirely — train data used for val callbacks
    test_pipeline:
        ...
```

---

### 7. Compatibility fixes (`exp_utils.py`)

The root-level `exp_utils.py` is a patched replacement for `openhands/core/exp_utils.py`, required for compatibility with PyTorch Lightning ≥ 1.8. The original used `LoggerCollection` and `logger_connector.configure_logger()`, both of which were removed in PL 1.8.

The replacement module provides the same public interface (`get_trainer`, `experiment_manager`) and behaviour:

- **`get_trainer(cfg)`** — constructs a `pl.Trainer` from the Hydra config and applies the experiment manager.
- **`experiment_manager(trainer, cfg)`** — attaches loggers, checkpointing, and early stopping to a trainer based on the `exp_manager` config block.
- **`configure_loggers(...)`** — creates `TensorBoardLogger` and/or `WandbLogger` and assigns them directly to `trainer.loggers` (the PL 1.8+ API).
- **`configure_checkpointing(trainer, cfg)`** — attaches a `ModelCheckpoint` callback.
- **`configure_early_stopping(trainer, cfg)`** — attaches an `EarlyStopping` callback.

The experiment scripts in `experiment_scripts/` import from this root-level file rather than the original library module.

---

## Installation

Create and activate the Conda environment:

```bash
conda env create -f environment_full.yaml
conda activate openhands
```

Then install the package in editable mode:

```bash
pip install -e .
```

---

## Datasets used

Please cite the respective datasets if you use them. See the original repository for licensing terms.

| Dataset  | Link |
|----------|------|
| AUTSL    | [Link](https://chalearnlap.cvc.uab.es/dataset/40/description/) |
| CSL      | [Link](http://home.ustc.edu.cn/~pjh/openresources/cslr-dataset-2015/index.html) |
| DEVISIGN | [Link](http://vipl.ict.ac.cn/homepage/ksl/data.html) |
| GSL      | [Link](https://vcl.iti.gr/dataset/gsl/) |
| INCLUDE  | [Link](https://sign-language.ai4bharat.org/#/INCLUDE) |
| LSA64    | [Link](http://facundoq.github.io/datasets/lsa64/) |
| WLASL    | [Link](https://dxli94.github.io/WLASL/) |
| ÍTM      | Icelandic Sign Language data (ÍTM) |

---

## Extraction of poses

Poses can be extracted from videos using the provided script:

```bash
python scripts/mediapipe_extract.py
```

---

## License

This project is released under the [Apache 2.0 license](LICENSE.txt), the same license as the original OpenHands library. Copyright for the original codebase remains with the AI4Bharat team.

---

## Citation

If you use this work, please cite the original OpenHands papers:

```bibtex
@misc{2021_openhands_slr_preprint,
      title={OpenHands: Making Sign Language Recognition Accessible with Pose-based Pretrained Models across Languages},
      author={Prem Selvaraj and Gokul NC and Pratyush Kumar and Mitesh Khapra},
      year={2021},
      eprint={2110.05877},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@inproceedings{
      nc2022addressing,
      title={Addressing Resource Scarcity across Sign Languages with Multilingual Pretraining and Unified-Vocabulary Datasets},
      author={Gokul NC and Manideep Ladi and Sumit Negi and Prem Selvaraj and Pratyush Kumar and Mitesh M Khapra},
      booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
      year={2022},
      url={https://openreview.net/forum?id=zBBmV-i84Go}
}
```
