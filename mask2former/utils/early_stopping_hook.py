#!/usr/bin/env python3
# LogPredMasksHook – dumps predicted semantic masks for 5 fixed samples
from detectron2.engine.hooks import HookBase
import numpy as np

class EarlyStoppingHook(HookBase):
    def __init__(self, period=1000, patience=5, min_delta=0.0, metric_name="sem_seg/mIoU", higher_is_better=True):
        super().__init__()
        self.period = period
        self.patience = patience
        self.min_delta = min_delta if higher_is_better else -min_delta
        self.metric_name = metric_name
        self.best_metric = -np.inf if higher_is_better else np.inf
        self.counter = 0
        self.should_stop = False
        self.last_eval_it = -1
        self.higher_is_better = higher_is_better

    def after_step(self):
        if self.should_stop:
            return

        it = self.trainer.iter
        if it % self.period == 0:  # Only check on evaluation steps
            output = self.trainer.storage.latest().get(self.metric_name, -np.inf)
            
            if not isinstance(output, tuple):
                return
            
            current_metric, last_eval_it = output
            
            if self.last_eval_it == last_eval_it:
                return
            
            metric_improved = (current_metric > self.best_metric + self.min_delta) if self.higher_is_better else (current_metric < self.best_metric - self.min_delta)

                        
            if metric_improved:
                self.best_metric = current_metric
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.should_stop = True
                    print(f"\nEarly stopping triggered! No improvement in {self.metric_name} for {self.patience} evaluations.")
                    print(f"Best {self.metric_name}: {self.best_metric:.4f}")
            
            self.last_eval_it = last_eval_it

    def after_train(self):
        if self.should_stop:
            raise StopIteration("Early stopping triggered")

