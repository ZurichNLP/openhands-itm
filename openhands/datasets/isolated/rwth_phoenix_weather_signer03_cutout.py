import os
import pandas as pd
from bs4 import BeautifulSoup
from .base import BaseIsolatedDataset

class RWTH_Phoenix_Signer03_Dataset(BaseIsolatedDataset):
    """
    German Isolated Sign language dataset from the paper:
    
    `RWTH-PHOENIX-Weather: A Large Vocabulary Sign Language Recognition and Translation Corpus. <https://www-i6.informatik.rwth-aachen.de/~forster/database-rwth-phoenix.php>`
    Signer03 cutout has been taken for the experiments :
    Image sequence - https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/rwth-phoenix-weather-signer03-cutout-images_20120820.tgz
    Anotations - https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/rwth-phoenix-weather-signer03-cutout_20120820.tgz
    """

    lang_code = "gsg"

    def read_glosses(self):
        s = set()
        with open(self.class_mappings_file_path , 'r') as f:
            data = f.read()
            # Bs_data = BeautifulSoup(data, "xml")
            Bs_data = BeautifulSoup(data, "lxml")
            glosses = Bs_data.find_all('orth')
            for gloss in glosses:
                s.add(gloss.text.strip())
        self.glosses = s


    def read_original_dataset(self):
        with open(self.split_file, 'r') as f:
            data = f.read()
            Bs_data = BeautifulSoup(data, "xml")
            filenames = Bs_data.find_all('recording')
            glosses = Bs_data.find_all('orth')

            unique_glosses = sorted(set(g.text.strip() for g in glosses))
            self.gloss_to_id = {gloss: idx for idx, gloss in enumerate(unique_glosses)}

            for filename, gloss in zip(filenames, glosses):
                key = gloss.text.strip()
                gloss_cat = self.gloss_to_id[key]
                instance_entry = filename.get('name'), gloss_cat
                self.data.append(instance_entry)
