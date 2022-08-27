import sys
import gc
import time
from torch.utils.data import DataLoader as PyTorchDataLoader
from torch.optim import Optimizer as PyTorchOptimizer
from torch.nn import Module as PyTorchModule
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp.grad_scaler import GradScaler
from tqdm import tqdm
import torch
import numpy as np

from utils.metrics import ArvnAccuMeter, ArvnAvgMeter



class WithPlaceholder(object):
    def __init__(self,name="1950641"):
        self.name=name

    def __enter__(self):
        return self

    def do_self(self):
        pass

    def __exit__(self,exc_type,exc_value,traceback):
        pass


class Epoch:
    r"""
    An experiment epoch
    """

    def __init__(self,
                 name: str,
                 model: PyTorchModule,
                 data_loader: PyTorchDataLoader,
                 optimizer: PyTorchOptimizer,
                 loss: PyTorchModule,
                 metrics: dict = None,
                 checkpoint_path: str = ".",
                 back_prop: bool = True,
                 device: str = "cuda",
                 save_checkpoints: bool = True,
                 auto_gc: bool = True,
                 dataloader_type: str = "segmentation",
                 metrics_mask = None,
                 scheduler: object = None,
                 backprop_gap: int = 1,
                 use_amp: bool = False
                 ):
        """
        Define a runner or testing epoch

        :param name: Model name.
        :param device: Training device. ['cuda','gpu','cuda0',...]
        :param model: Model
        :param data_loader: Data loader
        :param optimizer: Adopted optimizer
        :param loss:  Adopted loss function
        :param metrics: Set of functions to evaluate the experiment quality
        :param checkpoint_path: Path to place checkpoints
        :param back_prop: If true, do back-propagation
        """
        model = model.to(device)
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.loss = loss
        self.checkpoint_path = checkpoint_path
        self.back_prop = back_prop
        self.device = device
        self.name = name
        self.save_checkpoints = save_checkpoints
        self.auto_gc = auto_gc
        self.dltype = dataloader_type
        self.best_loss = 1e99
        self.backprop_gap = backprop_gap
        self.use_amp = use_amp
        if metrics is None:
            metrics = {}
        self.metrics = metrics
        self.metrics_mask = {}
        if metrics_mask is not None:
            self.metrics_mask = metrics_mask
        self.scheduler = scheduler
        for key in metrics.keys():
            if key in self.metrics_mask:
                self.metrics_mask[key] = metrics_mask[key]
            else:
                self.metrics_mask[key] = lambda *args: True

    def __get_metrics_dict__(self):
        ret = {}
        for key in self.metrics.keys():
            ret[key] = ArvnAvgMeter()
        return ret

    def __get_metrics_dict_org__(self):
        ret = {}
        for key in self.metrics.keys():
            ret[key] = 0.0
        return ret

    def __get_metrics_result_dict__(self, pred, mask):
        ret = {}
        for key in self.metrics.keys():
            if self.metrics_mask[key]():
                x = self.metrics[key](pred, mask)
            else:
                x = 0.0
            if type(x) is float:
                ret[key] = x
            elif type(x) is np.float64:
                ret[key] = np.float(x)
            else:
                ret[key] = x.detach().cpu().numpy()
            del x
        return ret

    def run(self, epoch_id=0):
        return self.run_impl(epoch_id)

    def run_impl(self, epoch_id):
        if self.back_prop:
            self.model.train()
            call_fn = torch.enable_grad
        else:
            self.model.eval()
            call_fn = torch.no_grad
        if self.device == "cuda" and self.use_amp:
            amp_fn = autocast
        else:
            amp_fn = WithPlaceholder
        scaler = GradScaler()
        total_metrics = self.__get_metrics_dict__()
        disp_metrics = self.__get_metrics_dict_org__()
        total_loss = 0
        counter = 0
        start_time = time.time()
        start_time_meter = ArvnAvgMeter()
        data_time_meter = ArvnAccuMeter()
        data_loader_size = len(self.data_loader)
        with tqdm(total=len(self.data_loader), file=sys.stdout, desc=self.name, ascii=True) as t:
            with call_fn():
                for b_idx, (image, coord) in enumerate(self.data_loader):
                    global_step = (epoch_id * data_loader_size + b_idx) // self.backprop_gap
                    counter += 1
                    image = image.to(self.device)
                    if isinstance(coord, list):
                        for i in range(len(coord)):
                            coord[i] = coord[i].to(self.device)
                    else:
                        coord = coord.to(self.device)
                    data_time_meter.add_value_raw(time.time() - start_time)
                    if self.back_prop:
                        if counter % self.backprop_gap == 0 or counter == len(self.data_loader) or counter == 1:
                            self.optimizer.zero_grad()
                    with amp_fn():
                        pred = self.model(image)
                        loss = self.loss(pred, coord) / self.backprop_gap
                    if self.back_prop:
                        if self.device == "cuda" and self.use_amp:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                    total_loss += loss.detach().cpu().numpy() * self.backprop_gap
                    if total_loss == np.nan:
                        raise Exception("Loss error: NaN")
                    if self.back_prop:
                        if counter % self.backprop_gap == 0 or counter == len(self.data_loader) or counter == 1:
                            if self.device == "cuda" and self.use_amp:
                                scaler.step(self.optimizer)
                            else:
                                self.optimizer.step()
                            if self.scheduler is not None:
                                self.scheduler.step(global_step)
                    with torch.no_grad():
                        iter_metrics = self.__get_metrics_result_dict__(pred, coord)
                    for keys in iter_metrics.keys():
                        if self.metrics_mask[keys]():
                            total_metrics[keys].add_value_raw(iter_metrics[keys])
                            disp_metrics[keys] = total_metrics[keys].get_value()
                    disp_metrics["loss"] = total_loss / counter
                    disp_metrics["data_time"] = data_time_meter.get_value()
                    t.update(1)
                    t.set_postfix(**disp_metrics)
                    del pred
                    start_time = time.time()
                if self.auto_gc:
                    gc.collect()
        if self.save_checkpoints:
            torch.save(self.model.state_dict(),
                       self.checkpoint_path + "/last.pth")
            torch.save(self.optimizer.state_dict(),
                       self.checkpoint_path + "/last.pth_optim.pth")
            if disp_metrics["loss"] < self.best_loss:
                self.best_loss = disp_metrics["loss"]
                torch.save(self.model.state_dict(),
                           self.checkpoint_path + "/best.pth")
                torch.save(self.optimizer.state_dict(),
                           self.checkpoint_path + "/best.pth_optim.pth")
        return disp_metrics
