from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_sem_seg
import os

DRIVABILIT_SEM_SEG_CATEGORIES = [
    {"name": "background", "id": 0, "color": (0, 0, 0)},
    {"name": "Dense Vegetation Barrier", "id": 1, "color": (0, 134, 134)},
    {"name": "Impassable Terrain", "id": 2, "color": (255, 33, 33)},
    {"name": "Paved Roadway", "id": 3, "color": (0, 125, 255)},
    {"name": "Short-Vegetation Terrain - Packed Ground", "id": 4, "color": (107, 202, 0)},
    {"name": "Unknown", "id": 5, "color": (255, 255, 0)},
    {"name": "Unstructured Drivable Ground", "id": 6, "color": (222, 127, 0)},
]

def _get_schwarmstudie_meta():
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing, so all ids are shifted by 1.
    stuff_ids = [k["id"] for k in DRIVABILIT_SEM_SEG_CATEGORIES]
    assert len(stuff_ids) == 7, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in DRIVABILIT_SEM_SEG_CATEGORIES]
    stuff_colors = [k["color"] for k in DRIVABILIT_SEM_SEG_CATEGORIES]
    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret

def register_all_schwarmstudie(root):
    root = os.path.join(root, "Drivability")
    meta = _get_schwarmstudie_meta()
    for name, dirname in [("train", "train"), ("val", "valid")]:
        image_dir = os.path.join(root, dirname, "images")
        gt_dir = os.path.join(root, dirname, "masks")
        name = f"Drivability_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=meta["stuff_classes"][:],
            stuff_colors=meta["stuff_colors"][:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,  # NOTE: gt is saved in 16-bit TIFF images,
            stuff_dataset_id_to_contiguous_id=meta["stuff_dataset_id_to_contiguous_id"],
        )
        
        print(f"Registered {name} with {len(DatasetCatalog.get(name))} images")

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_schwarmstudie(_root)
