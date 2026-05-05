import os
from .base import BaseIsolatedDataset
from .asllvd import ASLLVDDataset
from .autsl import AUTSLDataset
from .bosphorus22k import Bosphorus22kDataset
from .csl import CSLDataset
from .devisign import DeviSignDataset
from .gsl import GSLDataset
from .include import INCLUDEDataset
from .lsa64 import LSA64Dataset
from .msasl import MSASLDataset
from .rwth_phoenix_weather_signer03_cutout import RWTH_Phoenix_Signer03_Dataset
from .wlasl import WLASLDataset
from .itm import ITMDataset
import pandas as pd


class ConcatDataset(BaseIsolatedDataset):
    def __init__(self, datasets, unify_vocabulary=False, **kwargs):

        self.unify_vocabulary = unify_vocabulary
        self.datasets = []

        for dataset_cls_name, dataset_kwargs in datasets.items():
            kwargs_copy = dict(kwargs)
            kwargs_copy.update(dataset_kwargs)

            dataset_instance = globals()[dataset_cls_name](**kwargs_copy)
            self.datasets.append(dataset_instance)

        super().__init__(root_dir="", **kwargs)
        del self.datasets

        assert self.modality == "pose", "Only pose modality is currently supported for this dataset"

    def read_glosses(self):
        self.glosses = set()
        for dataset in self.datasets:
            for class_name in dataset.glosses:
                if self.unify_vocabulary:
                    if pd.isna(class_name):
                        continue
                    # Use .get() fallback: mapping only lists exceptions that need
                    # remapping; glosses already in normalized form are kept as-is
                    self.glosses.add(dataset.normalized_class_mappings.get(class_name, class_name))
                else:
                    self.glosses.add(f"{dataset.lang_code}__{class_name}")

        # TODO: Make the sequence agnostic to the order in which datasets are listed
        self.glosses = sorted(self.glosses)

    def read_original_dataset(self):
        for dataset in self.datasets:
            if dataset.only_metadata:
                continue

            # Build reverse mapping from gloss_to_id if id_to_gloss not available
            id_to_gloss = getattr(dataset, 'id_to_gloss', None)
            if id_to_gloss is None:
                id_to_gloss = {v: k for k, v in dataset.gloss_to_id.items()}

            for video_name, class_id in dataset.data:
                try:
                    class_name = id_to_gloss[class_id]
                except KeyError:
                    print(f"Warning: class_id '{class_id}' not found in id_to_gloss for dataset {dataset.__class__.__name__}. Skipping.")
                    continue
                if self.unify_vocabulary:
                    # Use .get() fallback: mapping only lists exceptions that need
                    # remapping; glosses already in normalized form are kept as-is
                    class_name = dataset.normalized_class_mappings.get(class_name, class_name)
                else:
                    class_name = f"{dataset.lang_code}__{class_name}"

                instance_entry = os.path.join(dataset.root_dir, video_name), self.gloss_to_id[class_name], dataset.lang_code, dataset.__class__.__name__
                self.data.append(instance_entry)
    
    def enumerate_data_files(self, dir):
        for dataset in self.datasets:
            for pose_path, _ in dataset.data:  # child data is (path, -1) tuples
                instance_entry = (
                    pose_path,
                    -1,
                    dataset.lang_code,
                    dataset.__class__.__name__,
                )
                self.data.append(instance_entry)