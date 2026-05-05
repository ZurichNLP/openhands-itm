import torch
import pytorch_lightning as pl
from tqdm import tqdm
import time

from ..core.data import DataModule
from ..models.loader import get_model
from sklearn.metrics import confusion_matrix
import numpy as np

# merge with the corresponding modules in the future release.
class InferenceModel(pl.LightningModule):
    """
    This will be the general interface for running the inference across models.
    Args:
        cfg (dict): configuration set.

    """
    def __init__(self, cfg, stage="test"):
        super().__init__()
        self.cfg = cfg
        self.datamodule = DataModule(cfg.data)
        self.datamodule.setup(stage=stage)

        self.model = self.create_model(cfg.model)
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if stage == "test":
            self.model.to(self._device).eval()

        # Build global mapping once if multilingual
        self._global_id_to_gloss = None
        self._global_gloss_to_id = None
        self._subdatasets_cache = None  # populated lazily by _get_subdatasets()
        if self._is_multilingual():
            self._global_id_to_gloss, self._global_gloss_to_id = self._build_global_mapping()

    # ------------------------------------------------------------------
    # Multilingual helpers
    # ------------------------------------------------------------------

    def _is_multilingual(self):
        """Returns True if multilingual is explicitly set in the test pipeline config."""
        return self.cfg.data.get("test_pipeline", {}).get("dataset", {}).get("multilingual", False)

    def _get_unify_vocabulary(self):
        """Returns the unify_vocabulary flag from the test pipeline config."""
        return self.cfg.data.get("test_pipeline", {}).get("dataset", {}).get("unify_vocabulary", False)

    def _get_subdatasets(self):
        """
        Reconstruct sub-datasets from config with only_metadata=True.
        ConcatDataset deletes self.datasets after __init__, so we re-instantiate
        each sub-dataset just enough to access glosses and normalized_class_mappings.
        Result is cached after the first call to avoid repeated expensive instantiation.
        """
        if self._subdatasets_cache is not None:
            return self._subdatasets_cache

        from ..datasets.isolated import (
            ASLLVDDataset, AUTSLDataset, GSLDataset, INCLUDEDataset,
            LSA64Dataset, MSASLDataset, RWTH_Phoenix_Signer03_Dataset,
            WLASLDataset, ITMDataset,
        )
        dataset_registry = {
            "ASLLVDDataset": ASLLVDDataset,
            "AUTSLDataset": AUTSLDataset,
            "GSLDataset": GSLDataset,
            "INCLUDEDataset": INCLUDEDataset,
            "LSA64Dataset": LSA64Dataset,
            "MSASLDataset": MSASLDataset,
            "RWTH_Phoenix_Signer03_Dataset": RWTH_Phoenix_Signer03_Dataset,
            "WLASLDataset": WLASLDataset,
            "ITMDataset": ITMDataset,
        }

        pipeline_cfg = self.cfg.data.get("test_pipeline", {})
        dataset_cfg = pipeline_cfg.get("dataset", {})

        # Keys that belong to ConcatDataset itself, not the sub-datasets
        concat_keys = {"_target_", "datasets", "unify_vocabulary", "multilingual",
                       "splits", "inference_mode", "modality"}
        shared_kwargs = {k: v for k, v in dataset_cfg.items() if k not in concat_keys}

        subdatasets = []
        for cls_name, cls_kwargs in dataset_cfg.get("datasets", {}).items():
            kwargs = {**shared_kwargs, **cls_kwargs, "only_metadata": True}
            subdatasets.append(dataset_registry[cls_name](**kwargs))

        self._subdatasets_cache = subdatasets
        return self._subdatasets_cache

    def _build_global_mapping(self):
        """
        Reconstruct the global gloss <-> id mapping exactly as ConcatDataset did
        during training:
          - prefix each gloss with lang_code__  (when unify_vocabulary=False)
          - normalize and deduplicate glosses (when unify_vocabulary=True)
          - sort the full set of glosses alphabetically
          - assign ids 0..N in that order
        """
        import pandas as pd
        unify_vocabulary = self._get_unify_vocabulary()
        glosses = set()
        for dataset in self._get_subdatasets():
            for class_name in dataset.glosses:
                if unify_vocabulary:
                    if pd.isna(class_name):
                        continue
                    glosses.add(dataset.normalized_class_mappings.get(class_name, class_name))
                else:
                    glosses.add(f"{dataset.lang_code}__{class_name}")

        glosses = sorted(glosses)  # must match ConcatDataset.read_glosses()
        gloss_to_id = {g: i for i, g in enumerate(glosses)}
        id_to_gloss = {i: g for i, g in enumerate(glosses)}
        return id_to_gloss, gloss_to_id

    def _local_to_global(self, gt_index, dataset_name):
        """
        Map a local label index from a sub-dataset into the global label space
        used by the trained model.

        Returns the global integer index, or -1 if the gloss cannot be found.
        """
        unify_vocabulary = self._get_unify_vocabulary()
        dataset = self._get_subdataset_by_name(dataset_name)
        id_to_gloss = getattr(dataset, 'id_to_gloss', None)
        if id_to_gloss is None:
            id_to_gloss = {v: k for k, v in dataset.gloss_to_id.items()}

        local_gloss = id_to_gloss.get(gt_index if isinstance(gt_index, int) else gt_index.item())
        if local_gloss is None:
            return -1

        if unify_vocabulary:
            prefixed = dataset.normalized_class_mappings.get(local_gloss, local_gloss)
        else:
            prefixed = f"{dataset.lang_code}__{local_gloss}"

        return self._global_gloss_to_id.get(prefixed, -1)

    def _get_subdataset_by_name(self, dataset_name):
        """Look up a sub-dataset by its class name."""
        for ds in self._get_subdatasets():
            if ds.__class__.__name__ == dataset_name:
                return ds
        raise ValueError(f"Sub-dataset '{dataset_name}' not found among test datasets.")

    def _resolve_label(self, pred_index):
        """
        Return a human-readable gloss for a predicted global index.
        Works for both multilingual (global mapping) and monolingual (dataset mapping).
        """
        idx = pred_index.item() if isinstance(pred_index, torch.Tensor) else pred_index
        if self._global_id_to_gloss is not None:
            return self._global_id_to_gloss.get(idx, f"<unknown:{idx}>")
        return self.datamodule.test_dataloader().dataset.id_to_gloss.get(idx, f"<unknown:{idx}>")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def create_model(self, cfg):
        """Creates and returns the model object based on the config."""
        return get_model(cfg, self.datamodule.in_channels, self.datamodule.num_class)

    def forward(self, x):
        """Forward propagates the inputs and returns the model output."""
        return self.model(x)

    def init_from_checkpoint_if_available(self, map_location=torch.device("cpu")):
        """Initializes the pretrained weights if the cfg has a pretrained parameter."""
        if "pretrained" not in self.cfg.keys():
            return

        ckpt_path = self.cfg["pretrained"]
        print(f"Loading checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=map_location)

        pretrained_state = ckpt["state_dict"]
        model_state = self.state_dict()
        # strict=False skips missing/unexpected keys but still errors on shape mismatches;
        # filter those out manually so the encoder loads cleanly when the head differs.
        compatible_state = {
            k: v for k, v in pretrained_state.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        skipped = [k for k in pretrained_state if k not in compatible_state]
        if skipped:
            print(f"Skipped {len(skipped)} mismatched/missing key(s): {skipped}")
        self.load_state_dict(compatible_state, strict=False)
        del ckpt

    # ------------------------------------------------------------------
    # Inference / evaluation
    # ------------------------------------------------------------------

    def test_inference(self):
        """
        Runs inference over the test dataloader and prints predicted labels.
        Handles both monolingual and multilingual label spaces.
        """
        dataloader = self.datamodule.test_dataloader()
        total_time_taken, num_steps = 0.0, 0

        for batch in dataloader:
            start_time = time.time()
            y_hat = self.model(batch["frames"].to(self._device)).cpu()
            class_indices = torch.argmax(y_hat, dim=-1)

            for i, pred_index in enumerate(class_indices):
                label = self._resolve_label(pred_index)
                filename = batch["files"][i]
                print(f"{label}:\t{filename}")

            total_time_taken += time.time() - start_time
            num_steps += 1

        print(f"Avg time per iteration: {total_time_taken * 1000.0 / num_steps} ms")

    def compute_test_accuracy(self):
        """
        Computes per-dataset and average class-wise accuracy over the test dataloader.
        For multilingual models, ground-truth labels are remapped from each sub-dataset's
        local index space into the global index space used by the trained model.
        """
        assert not self.datamodule.test_dataset.inference_mode

        dataloader = self.datamodule.test_dataloader()
        dataset_scores, class_scores = {}, {}

        for batch_idx, batch in tqdm(enumerate(dataloader), unit="batch"):
            y_hat = self.model(batch["frames"].to(self._device)).cpu()
            #if batch_idx == 0:
            #    print(f"y_hat shape: {y_hat.shape}")
            #    print(f"num_class from datamodule: {self.datamodule.num_class}")
            class_indices = torch.argmax(y_hat, dim=-1)

            for i, (pred_index, gt_index) in enumerate(zip(class_indices, batch["labels"])):
                dataset_name = batch["dataset_names"][i]
                global_gt_index = gt_index.item()

                score = pred_index.item() == global_gt_index

                # Uncomment to print predicted label per file (like test_inference):
                label = self._resolve_label(pred_index)
                filename = batch["files"][i]
                print(f"{label}:\t{filename}\t{score}")

                if dataset_name not in dataset_scores:
                    dataset_scores[dataset_name] = []
                dataset_scores[dataset_name].append(score)

                if global_gt_index not in class_scores:
                    class_scores[global_gt_index] = []
                class_scores[global_gt_index].append(score)

        for dataset_name, score_array in dataset_scores.items():
            n = len(score_array)
            acc = sum(score_array) / n
            print(f"Accuracy for {n} samples in {dataset_name}: {acc * 100:.4f}%")

        classwise_accuracies = {
            cls: sum(s) / len(s) for cls, s in class_scores.items()
        }
        avg_classwise_accuracy = sum(classwise_accuracies.values()) / len(classwise_accuracies)
        print(f"Average of class-wise accuracies: {avg_classwise_accuracy * 100:.4f}%")

    def compute_test_avg_class_accuracy(self):
        """
        Computes overall average class accuracy via confusion matrix.
        For multilingual models, ground-truth labels are remapped from each sub-dataset's
        local index space into the global index space used by the trained model.
        """
        assert not self.datamodule.test_dataset.inference_mode

        dataloader = self.datamodule.test_dataloader()
        all_pred_indices = []
        all_gt_indices = []

        for batch_idx, batch in tqdm(enumerate(dataloader), unit="batch"):
            y_hat = self.model(batch["frames"].to(self._device)).cpu()
            class_indices = torch.argmax(y_hat, dim=-1)

            for i, (pred_index, gt_index) in enumerate(zip(class_indices, batch["labels"])):
                dataset_name = batch["dataset_names"][i]
                global_gt_index = gt_index.item()

                all_pred_indices.append(pred_index.item())
                all_gt_indices.append(global_gt_index)

        cm = confusion_matrix(np.array(all_gt_indices), np.array(all_pred_indices))
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        avg_class_acc = np.mean(cm.diagonal())
        print(f"Average Class Accuracy for {len(all_gt_indices)} samples: {avg_class_acc * 100:.4f}%")
        
        
# import torch
# import pytorch_lightning as pl
# from tqdm import tqdm
# import time

# from ..core.data import DataModule
# from ..models.loader import get_model
# from sklearn.metrics import confusion_matrix
# import numpy as np

# # merge with the corresponding modules in the future release.
# class InferenceModel(pl.LightningModule):
#     """
#     This will be the general interface for running the inference across models.
#     Args:
#         cfg (dict): configuration set.

#     """
#     def __init__(self, cfg, stage="test"):
#         super().__init__()
#         self.cfg = cfg
#         self.datamodule = DataModule(cfg.data)
#         self.datamodule.setup(stage=stage)

#         self.model = self.create_model(cfg.model)
#         self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         if stage == "test":
#             self.model.to(self._device).eval()
    
#     def create_model(self, cfg):
#         """
#         Creates and returns the model object based on the config.
#         """
#         return get_model(cfg, self.datamodule.in_channels, self.datamodule.num_class)
    
#     def forward(self, x):
#         """
#         Forward propagates the inputs and returns the model output.
#         """
#         return self.model(x)
    
#     def init_from_checkpoint_if_available(self, map_location=torch.device("cpu")):
#         """
#         Intializes the pretrained weights if the ``cfg`` has ``pretrained`` parameter.
#         """
#         if "pretrained" not in self.cfg.keys():
#             return

#         ckpt_path = self.cfg["pretrained"]
#         print(f"Loading checkpoint from: {ckpt_path}")
#         ckpt = torch.load(ckpt_path, map_location=map_location)
#         self.load_state_dict(ckpt["state_dict"], strict=False)
#         del ckpt
    
#     def test_inference(self):
#         """
#         Calculates the time taken for inference for all the batches in the test dataloader.
#         """
#         # TODO: Write output to a csv
#         dataloader = self.datamodule.test_dataloader()
#         total_time_taken, num_steps = 0.0, 0

#         for batch in dataloader:
#             start_time = time.time()
#             y_hat = self.model(batch["frames"].to(self._device)).cpu()

#             class_indices = torch.argmax(y_hat, dim=-1)

#             for i, pred_index in enumerate(class_indices):
#                 # label = self.datamodule.test_dataset.id_to_gloss[pred_index]
#                 label = dataloader.dataset.id_to_gloss[pred_index.item()]
#                 filename = batch["files"][i]
#                 print(f"{label}:\t{filename}")
            
#             total_time_taken += time.time() - start_time
#             num_steps += 1
        
#         print(f"Avg time per iteration: {total_time_taken*1000.0/num_steps} ms")

#     def compute_test_accuracy(self):
#         """
#         Computes the accuracy for the test dataloader.
#         """
#         # Ensure labels are loaded
#         assert not self.datamodule.test_dataset.inference_mode
#         # TODO: Write output to a csv
#         dataloader = self.datamodule.test_dataloader()
#         dataset_scores, class_scores = {}, {}
#         for batch_idx, batch in tqdm(enumerate(dataloader), unit="batch"):
#             y_hat = self.model(batch["frames"].to(self._device)).cpu()
#             class_indices = torch.argmax(y_hat, dim=-1)
#             for i, (pred_index, gt_index) in enumerate(zip(class_indices, batch["labels"])):

#                 dataset_name = batch["dataset_names"][i]
#                 score = pred_index == gt_index
                
#                 if dataset_name not in dataset_scores:
#                     dataset_scores[dataset_name] = []
#                 dataset_scores[dataset_name].append(score)

#                 if gt_index not in class_scores:
#                     class_scores[gt_index] = []
#                 class_scores[gt_index].append(score)
        
        
#         for dataset_name, score_array in dataset_scores.items():
#             dataset_accuracy = sum(score_array)/len(score_array)
#             print(f"Accuracy for {len(score_array)} samples in {dataset_name}: {dataset_accuracy*100}%")


#         classwise_accuracies = {class_index: sum(scores)/len(scores) for class_index, scores in class_scores.items()}
#         avg_classwise_accuracies = sum(classwise_accuracies.values()) / len(classwise_accuracies)

#         print(f"Average of class-wise accuracies: {avg_classwise_accuracies*100}%")
    
#     def compute_test_avg_class_accuracy(self):
#         """
#         Computes the accuracy for the test dataloader.
#         """
#         #Ensure labels are loaded
#         assert not self.datamodule.test_dataset.inference_mode
#         # TODO: Write output to a csv
#         dataloader = self.datamodule.test_dataloader()
#         scores = []
#         all_class_indices=[]
#         all_batch_labels=[]
#         for batch_idx, batch in tqdm(enumerate(dataloader),unit="batch"):
#             y_hat = self.model(batch["frames"].to(self._device)).cpu()
#             class_indices = torch.argmax(y_hat, dim=-1)

#             for i in range(len(batch["labels"])):
#                 all_batch_labels.append(batch["labels"][i])
#                 all_class_indices.append(class_indices[i])
#             for pred_index, gt_index in zip(class_indices, batch["labels"]):
#                 scores.append(pred_index == gt_index)
#         cm = confusion_matrix(np.array(all_batch_labels), np.array(all_class_indices))
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print(f"Average Class Accuracy for {len(all_batch_labels)} samples: {np.mean(cm.diagonal())*100}%")
