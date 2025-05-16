# Mask2Former: Masked-attention Mask Transformer for Universal Image Segmentation (CVPR 2022)

[Bowen Cheng](https://bowenc0221.github.io/), [Ishan Misra](https://imisra.github.io/), [Alexander G. Schwing](https://alexander-schwing.de/), [Alexander Kirillov](https://alexander-kirillov.github.io/), [Rohit Girdhar](https://rohitgirdhar.github.io/)

[[`arXiv`](https://arxiv.org/abs/2112.01527)] [[`Project`](https://bowenc0221.github.io/mask2former)] [[`BibTeX`](#CitingMask2Former)]

<div align="center">
  <img src="https://bowenc0221.github.io/images/maskformerv2_teaser.png" width="100%" height="100%"/>
</div><br/>

### Features
* A single architecture for panoptic, instance and semantic segmentation.
* Support major segmentation datasets: ADE20K, Cityscapes, COCO, Mapillary Vistas.

## Updates
* Add Google Colab demo.
* Video instance segmentation is now supported! Please check our [tech report](https://arxiv.org/abs/2112.10764) for more details.

## Installation

See [installation instructions](INSTALL.md).

## Getting Started

See [Preparing Datasets for Mask2Former](datasets/README.md).

See [Getting Started with Mask2Former](GETTING_STARTED.md).

Run our demo using Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uIWE5KbGFSjrxey2aRd5pWkKNY1_SaNq)

Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the Web Demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/Mask2Former)

Replicate web demo and docker image is available here: [![Replicate](https://replicate.com/facebookresearch/mask2former/badge)](https://replicate.com/facebookresearch/mask2former)

## Training a New Semantic Segmentation Model
### 0. Set Dataset Path
First, let `detectron` know where your data lies.
```
export DETECTRON2_DATASETS=path/to/datasets
```
### 1. Dataset Preparation
Organize your dataset in the following structure:
```
dataset_root/
├── train/
│   ├── images/
│   │   └── *.jpg
│   └── masks/
│       └── *.png
└── valid/
    ├── images/
    │   └── *.jpg
    └── masks/
        └── *.png
```

### 2. Dataset Registration
1. Create a new file in `mask2former/data/datasets/register_your_dataset.py`:
```python
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_sem_seg
import os

# Define your categories
YOUR_DATASET_CATEGORIES = [
    {"name": "background", "id": 0, "color": (0, 0, 0)},
    {"name": "class1", "id": 1, "color": (255, 0, 0)},
    # Add more classes as needed
]

def _get_dataset_meta():
    stuff_ids = [k["id"] for k in YOUR_DATASET_CATEGORIES]
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in YOUR_DATASET_CATEGORIES]
    stuff_colors = [k["color"] for k in YOUR_DATASET_CATEGORIES]
    return {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }

def register_all_your_dataset(root):
    root = os.path.join(root, "your_dataset_name")
    meta = _get_dataset_meta()
    for name, dirname in [("train", "train"), ("val", "valid")]:
        image_dir = os.path.join(root, dirname, "images")
        gt_dir = os.path.join(root, dirname, "masks")
        name = f"your_dataset_name_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=meta["stuff_classes"][:],
            stuff_colors=meta["stuff_colors"][:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            stuff_dataset_id_to_contiguous_id=meta["stuff_dataset_id_to_contiguous_id"],
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_schwarmstudie(_root)

```

2. Add your dataset registration to `mask2former/data/datasets/__init__.py`:
```python
from . import (
    # ... existing imports ...
    register_your_dataset,
)
```

### 3. Dataset Mapper
Create a new file in `mask2former/data/dataset_mappers/your_dataset_mapper.py`:
```python
from mask2former.data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)
import detectron2.data.transforms as T

class YourDatasetMapper(MaskFormerSemanticDatasetMapper):
    """
    Custom dataset mapper for your dataset
    """
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)
        # Customize augmentations as needed
        self.augmentations = [
            T.Resize((640, 640)),
            T.RandomRotation(angle=[-180, 180], sample_style="range", expand=False),
            T.RandomCrop("absolute", (640, 640)),
        ]
```

### 4. Configuration
Create a new config file in `configs/your_dataset/your_config.yaml`:
```yaml
_BASE_: "../mask2former/maskformer_swin_tiny_IN21k_384_bs16_160k.yaml"
MODEL:
  MASK_FORMER:
    NUM_QUERIES: 100
    TRANSFORMER_DECODER_NAME: "MultiScaleMaskedTransformerDecoder"
    TRANSFORMER_IN_FEATURE: "res5"
    COMMON_STRIDE: 32
    NUM_CLASSES: <number_of_your_classes>
  PIXEL_DECODER:
    NAME: "BasePixelDecoder"
    IN_FEATURES: ["res2", "res3", "res4", "res5"]
    COMMON_STRIDE: 4
    CONVS_DIM: 256
    MASK_DIM: 256
    NORM: "GN"
    NUM_GROUPS: 32
    USE_CHECKPOINT: False
    IGNORE_VALUE: 255
DATASETS:
  TRAIN: ("your_dataset_name_train",)
  TEST: ("your_dataset_name_val",)
SOLVER:
  IMS_PER_BATCH: 16
  BASE_LR: 0.0001
  MAX_ITER: 90000
  STEPS: (60000, 80000)
  GAMMA: 0.1
  WARMUP_FACTOR: 1.0 / 1000
  WARMUP_ITERS: 1000
  WARMUP_METHOD: "linear"
  CHECKPOINT_PERIOD: 5000
  EVAL_PERIOD: 5000
INPUT:
  MIN_SIZE_TRAIN: (640, 672, 704, 736, 768, 800)
  MAX_SIZE_TRAIN: 1333
  MIN_SIZE_TEST: 800
  MAX_SIZE_TEST: 1333
  MASK_FORMAT: "polygon"
DATALOADER:
  NUM_WORKERS: 2
  ASPECT_RATIO_GROUPING: True
  REPEAT_THRESHOLD: 0.0
  FILTER_EMPTY_ANNOTATIONS: True
```

### 5. Training
Start training with:
```bash
python train_net.py \
    --config-file configs/your_dataset/your_config.yaml \
    --num-gpus 8 \
    OUTPUT_DIR output/your_dataset
```

### Important Notes:
1. Make sure your mask images are single-channel PNG files where pixel values correspond to class IDs
2. The background class should have ID 0
3. Set `ignore_label=255` for pixels you want to ignore during training
4. Adjust the configuration parameters (learning rate, batch size, etc.) based on your dataset size and available GPU memory
5. The dataset mapper's augmentations can be customized based on your needs
6. Make sure to update the number of classes in the config file to match your dataset

## Advanced usage

See [Advanced Usage of Mask2Former](ADVANCED_USAGE.md).

## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the [Mask2Former Model Zoo](MODEL_ZOO.md).

## License

Shield: [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The majority of Mask2Former is licensed under a [MIT License](LICENSE).


However portions of the project are available under separate license terms: Swin-Transformer-Semantic-Segmentation is licensed under the [MIT license](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/LICENSE), Deformable-DETR is licensed under the [Apache-2.0 License](https://github.com/fundamentalvision/Deformable-DETR/blob/main/LICENSE).

## <a name="CitingMask2Former"></a>Citing Mask2Former

If you use Mask2Former in your research or wish to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md), please use the following BibTeX entry.

```BibTeX
@inproceedings{cheng2021mask2former,
  title={Masked-attention Mask Transformer for Universal Image Segmentation},
  author={Bowen Cheng and Ishan Misra and Alexander G. Schwing and Alexander Kirillov and Rohit Girdhar},
  journal={CVPR},
  year={2022}
}
```

If you find the code useful, please also consider the following BibTeX entry.

```BibTeX
@inproceedings{cheng2021maskformer,
  title={Per-Pixel Classification is Not All You Need for Semantic Segmentation},
  author={Bowen Cheng and Alexander G. Schwing and Alexander Kirillov},
  journal={NeurIPS},
  year={2021}
}
```

## Acknowledgement

Code is largely based on MaskFormer (https://github.com/facebookresearch/MaskFormer).
