import os
import json
from .base import BaseIsolatedDataset
from ..data_readers import load_frames_from_video

class ITMDataset(BaseIsolatedDataset):
    """
    ÍTM Dataset
    """

    lang_code = "icl"

    def read_glosses(self):
        with open(self.split_file, "r") as f:
            self.content = json.load(f)
        self.glosses = sorted([gloss_entry["word_label"] for gloss_entry in self.content])

    def read_original_dataset(self):
        for gloss_entry in self.content:
            gloss, instances = gloss_entry["word_label"], gloss_entry["instances"]

            for instance in instances:
                if instance["split"] not in self.splits:
                    continue

                instance_entry = instance["video_id"], self.gloss_to_id[gloss]
                self.data.append(instance_entry)

    def read_video_data(self, index):
        video_name, label = self.data[index]
        video_path = os.path.join(self.root_dir, video_name + ".mp4")
        imgs = load_frames_from_video(video_path)
        return imgs, label, video_name
