#!/usr/bin/env python3
# LogPredMasksHook – dumps predicted semantic masks for 5 fixed samples
from detectron2.engine.hooks import HookBase
from detectron2.utils import visualizer as vis_utils
from detectron2.data import MetadataCatalog, DatasetCatalog
import detectron2.data.transforms as T
import torch, cv2, numpy as np, pathlib
import random

class LogPredMasksHook(HookBase):
    def __init__(self, cfg, dataset_name, num_samples=5, period=1000, out_dir="mask_vis", inf_size=(640, 640)):
        self.cfg = cfg
        self.period = period
        self.out = pathlib.Path(cfg.OUTPUT_DIR) / out_dir
        self.out.mkdir(parents=True, exist_ok=True)

        # pick N distinct items once
        ds = DatasetCatalog.get(dataset_name)
        self.samples = random.sample(ds, num_samples)

        self.metadata = MetadataCatalog.get(dataset_name)
        self.model = None                             # filled on first call
        self.inf_size = inf_size
    def _run_once(self, it):
        was_training = self.model.training     # remember
        self.model.eval()
        resize = T.Resize(self.inf_size)
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            for idx, record in enumerate(self.samples):
                bgr = cv2.imread(record["file_name"])
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                aug_in  = T.AugInput(rgb)
                T.AugmentationList([resize])(aug_in)       # ← apply resize
                img_resized = aug_in.image.copy()                 # HWC uint8
                h, w = img_resized.shape[:2]

                inputs = [{"image": torch.from_numpy(img_resized).permute(2,0,1).cuda(),
                           "height": h, "width": w}]
                pred = self.model(inputs)[0]["sem_seg"].argmax(0).cpu().numpy()

                vis = vis_utils.Visualizer(img_resized, self.metadata).draw_sem_seg(pred, alpha=0.5).get_image()
                cv2.imwrite(str(self.out / f"iter{it:07d}_img{idx}.png"), vis[:, :, ::-1])
        if was_training:
            self.model.train()
    def after_step(self):
        it = self.trainer.iter
        if it % self.period == 0:
            if self.model is None:
                self.model = self.trainer.model
            self._run_once(it)
