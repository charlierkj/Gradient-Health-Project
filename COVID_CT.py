"""UCSD-AI4H/COVID-CT dataset:
   https://github.com/UCSD-AI4H/COVID-CT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow_datasets.public_api as tfds

import os
import zipfile
import git

import numpy as np

_CITATION = """\
@article{zhao2020COVID-CT-Dataset,
  title={COVID-CT-Dataset: a CT scan dataset about COVID-19},
  author={Zhao, Jinyu and Zhang Yichen and He, Xuehai and Xie, Pengtao},
  journal={arXiv preprint arXiv:2003.13865},
  year={2020}
}
"""

_DESCRIPTION = """\
The COVID-CT Dataset has 349 CT images containing clinical findings of COVID-19.
The images are collected from COVID19-related papers from medRxiv, bioRxiv, NEJM,
JAMA, Lancet, etc. CTs containing COVID-19 abnormalities are selected by reading
the figure captions in the papers. All copyrights of the data belong to the authors
and publishers of these papers.
"""

_URL = "https://github.com/UCSD-AI4H/COVID-CT/zipball/2d51929434b7ba91bdf7a983cd8131f231ea5bee"

class CovidCt(tfds.core.GeneratorBasedBuilder):
  """COVID-CT dataset."""

  VERSION = tfds.core.Version('1.0.0')
 
  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            "image": tfds.features.Image(),
            "label": tfds.features.ClassLabel(
                names=["COVID", "NonCOVID"]),
        }),
        supervised_keys=("image", "label"),
        homepage='https://github.com/UCSD-AI4H/COVID-CT',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""

    #extracted_path = dl_manager.download_and_extract(_URL)
    #extracted_path = "/home/kejia/tmp/COVID-CT"
    gitpath = os.path.expanduser('~')
    extracted_path = os.path.join(gitpath, "COVID-CT")
    print(extracted_path)
    if not os.path.exists(extracted_path):
        git.Git(gitpath).clone("https://github.com/UCSD-AI4H/COVID-CT.git")

    imgs_path = os.path.join(extracted_path, "Images-processed")
    label_path = os.path.join(extracted_path, "Data-split")

    with zipfile.ZipFile(os.path.join(extracted_path, "Images-processed/CT_COVID.zip"), 'r') as zip_pos:
        zip_pos.extractall(imgs_path)
    with zipfile.ZipFile(os.path.join(extracted_path, "Images-processed/CT_NonCOVID.zip"), 'r') as zip_neg:
        zip_neg.extractall(imgs_path)

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                "image_dir_path": {
                    'COVID': os.path.join(imgs_path, "CT_COVID"),
                    'NonCOVID': os.path.join(imgs_path, "CT_NonCOVID")},
                "labels": {
                    'COVID': os.path.join(label_path, "COVID/trainCT_COVID.txt"),
                    'NonCOVID': os.path.join(label_path, "NonCOVID/trainCT_NonCOVID.txt")}
                },
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={
                "image_dir_path": {
                    'COVID': os.path.join(imgs_path, "CT_COVID"),
                    'NonCOVID': os.path.join(imgs_path, "CT_NonCOVID")},
                "labels": {
                    'COVID': os.path.join(label_path, "COVID/valCT_COVID.txt"),
                    'NonCOVID': os.path.join(label_path, "NonCOVID/valCT_NonCOVID.txt")}
                },
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs={
                "image_dir_path": {
                    'COVID': os.path.join(imgs_path, "CT_COVID"),
                    'NonCOVID': os.path.join(imgs_path, "CT_NonCOVID")},
                "labels": {
                    'COVID': os.path.join(label_path, "COVID/testCT_COVID.txt"),
                    'NonCOVID': os.path.join(label_path, "NonCOVID/testCT_NonCOVID.txt")}
                },
        ),
    ]
        
  def _generate_examples(self, image_dir_path, labels):
    """Yields examples."""

    for (l, cls) in enumerate(['NonCOVID', 'COVID']):
        data_path = image_dir_path[cls]
        data_files = np.loadtxt(labels[cls], dtype=np.str)
        for data_file in data_files:
            yield data_file, {
                    "image": os.path.join(data_path, data_file),
                    "label": l,
                    }


