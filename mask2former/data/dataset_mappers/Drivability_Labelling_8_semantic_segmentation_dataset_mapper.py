from mask2former.data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)
import detectron2.data.transforms as T

class DrivabilityLabelling8SemanticSegmentationDatasetMapper(MaskFormerSemanticDatasetMapper):
    """
    Mask2Former semantic mapper with 640×640 + full 360° rotation.
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)
        # override the default LSJ pipeline with your 3 augments
        self.augmentations = [
            T.Resize((640, 640)),
            T.RandomRotation(angle=[-180, 180], sample_style="range", expand=False),
            T.RandomCrop("absolute", (640, 640)),
        ]
