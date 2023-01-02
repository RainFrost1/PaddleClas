# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import platform
import paddle
import paddle.distributed as dist
from visualdl import LogWriter
from paddle import nn
import numpy as np
import random
from abc import ABC, abstractmethod

from ppcls.utils.misc import AverageMeter
from ppcls.utils import logger
from ppcls.utils.logger import init_logger
from ppcls.utils.config import print_config
from ppcls.data import build_dataloader
from ppcls.arch import build_model, RecModel, DistillationModel, TheseusLayer
from ppcls.arch import apply_to_static
from ppcls.loss import build_loss
from ppcls.metric import build_metrics
from ppcls.optimizer import build_optimizer
from ppcls.utils.ema import ExponentialMovingAverage
from ppcls.utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url
from ppcls.utils.save_load import init_model
from ppcls.utils import save_load

from ppcls.data.utils.get_image_list import get_image_list
from ppcls.data.postprocess import build_postprocess
from ppcls.data import create_operators
from ppcls.engine import train as train_method
from ppcls.engine.train.utils import type_name
from ppcls.engine import evaluation
from ppcls.arch.gears.identity_head import IdentityHead


class BaseEngine(ABC):
    def __init__(self, config, mode="train"):
        assert mode in ["train", "eval", "infer", "export"]
        self.mode = mode
        self.config = config
        self.eval_mode = self.config["Global"].get("eval_mode", "classification").lower()
        self.train_mode = self.config["Global"].get("train_mode", "classification").lower()

        # init logger
        self.output_dir = self.config['Global']['output_dir']
        log_file = os.path.join(self.output_dir, self.config["Arch"]["name"],
                                f"{mode}.log")
        init_logger(log_file=log_file)
        print_config(config)

        # for visualdl
        self.vdl_writer = None
        if self.config['Global'][
                'use_visualdl'] and mode == "train" and dist.get_rank() == 0:
            vdl_writer_path = os.path.join(self.output_dir, "vdl")
            if not os.path.exists(vdl_writer_path):
                os.makedirs(vdl_writer_path)
            self.vdl_writer = LogWriter(logdir=vdl_writer_path)

        # set device
        assert self.config["Global"][
            "device"] in ["cpu", "gpu", "xpu", "npu", "mlu", "ascend"]
        self.device = paddle.set_device(self.config["Global"]["device"])
        logger.info('train with paddle {} and device {}'.format(
            paddle.__version__, self.device))

        # init base componet
        self.model, self.model_ema = None, None
        self.train_dataloader, self.eval_dataloader = None, None
        self.train_metric_func, self.eval_metric_func = None, None
        self.train_loss_func, self.eval_metric_func = None, None
        self.optimizer, self.lr_sch = None, None

        # global iter counter
        self.global_step = 0

    def build_component(self, build_dataloader=True, build_model=True, build_loss=True, build_optimizer=True, build_metrics=True, build_process=True):
        # gradient accumulation
        self.update_freq = self.config["Global"].get("update_freq", 1)

        # build dataloader
        self.use_dali = self.config['Global'].get("use_dali", False)
        if self.mode == 'train':
            self.train_dataloader = build_dataloader(
                self.config["DataLoader"], "Train", self.device, self.use_dali)

            self.iter_per_epoch = len(
                self.train_dataloader) - 1 if platform.system(
                ) == "Windows" else len(self.train_dataloader)
            if self.config["Global"].get("iter_per_epoch", None):
                # set max iteration per epoch mannualy, when training by iteration(s), such as XBM, FixMatch.
                self.iter_per_epoch = self.config["Global"].get(
                    "iter_per_epoch")
            self.iter_per_epoch = self.iter_per_epoch // self.update_freq * self.update_freq

        if self.mode == "eval" or (self.mode == "train" and
                                   self.config["Global"]["eval_during_train"]):
            self.eval_dataloader = build_dataloader(
                self.config["DataLoader"], "Eval", self.device,
                self.use_dali)
           
        # build loss
        if self.mode == "train":
            label_loss_info = self.config["Loss"]["Train"]
            self.train_loss_func = build_loss(label_loss_info)

        if self.mode == "eval" or (self.mode == "train" and
                                   self.config["Global"]["eval_during_train"]):
            loss_config = self.config.get("Loss", None)
            if loss_config is not None and loss_config.get("Eval", None) is not None:
                self.eval_loss_func = build_loss(loss_config.get("Eval"))

        # build metric
        if self.mode == 'train' and "Metric" in self.config and "Train" in self.config[
                "Metric"] and self.config["Metric"]["Train"]:
            metric_config = self.config["Metric"]["Train"]
            self.train_metric_func = build_metrics(metric_config)

        if self.mode == "eval" or (self.mode == "train" and
                                   self.config["Global"]["eval_during_train"]):
            self.eval_metric_func = build_metrics(metric_config)

        # build model
        self.model = build_model(self.config, self.mode)
        # set @to_static for benchmark, skip this by default.
        apply_to_static(self.config, self.model)

        # load_pretrain
        if self.config["Global"]["pretrained_model"] is not None:
            if self.config["Global"]["pretrained_model"].startswith("http"):
                load_dygraph_pretrain_from_url(
                    [self.model, getattr(self, 'train_loss_func', None)],
                    self.config["Global"]["pretrained_model"])
            else:
                load_dygraph_pretrain(
                    [self.model, getattr(self, 'train_loss_func', None)],
                    self.config["Global"]["pretrained_model"])

        # build optimizer
        if self.mode == 'train':
            self.optimizer, self.lr_sch = build_optimizer(
                self.config["Optimizer"], self.config["Global"]["epochs"],
                self.iter_per_epoch // self.update_freq,
                [self.model, self.train_loss_func])

        # build EMA model
        self.ema = "EMA" in self.config and self.mode == "train"
        if self.ema:
            self.model_ema = ExponentialMovingAverage(
                self.model, self.config['EMA'].get("decay", 0.9999))

        # build postprocess for infer
        if self.mode == 'infer':
            self.preprocess_func = create_operators(self.config["Infer"][
                "transforms"])
            self.postprocess_func = build_postprocess(self.config["Infer"][
                "PostProcess"])

    def set_train_attribute(self):
        # set seed
        self.seed = self.config["Global"].get("seed", None)
        if self.seed or self.seed == 0:
            assert isinstance(self.seed, int), "The 'seed' must be a integer!"
            paddle.seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

        # for distributed
        if self.config["Global"]["distributed"]:
            dist.init_parallel_env()
            assert self.model, "Please build model before using paddle.DataParallel."
            assert self.train_loss_func, "Please build train_loss before using paddle.DataParallel"
            self.model = paddle.DataParallel(self.model)
            if self.mode == 'train' and len(self.train_loss_func.parameters()) > 0:
                self.train_loss_func = paddle.DataParallel(
                    self.train_loss_func)

            # set different seed in different GPU manually in distributed environment
            if self.seed is None:
                logger.warning(
                    "The random seed cannot be None in a distributed environment. Global.seed has been set to 42 by default"
                )
                self.config["Global"]["seed"] = self.seed = 42
            logger.info(
                f"Set random seed to ({int(self.seed)} + $PADDLE_TRAINER_ID) for different trainer"
            )
            paddle.seed(int(self.seed) + dist.get_rank())
            np.random.seed(int(self.seed) + dist.get_rank())
            random.seed(int(self.seed) + dist.get_rank())
        
        # AMP training and evaluating
        self.amp = "AMP" in self.config and self.config["AMP"] is not None
        self.amp_eval = False
        if self.amp:
            assert self.model, "Please build model first when using AMP training"
            assert self.optimizer, "Please build optimizer first when using AMP training"
            AMP_RELATED_FLAGS_SETTING = {'FLAGS_max_inplace_grad_add': 8, }
            if paddle.is_compiled_with_cuda():
                AMP_RELATED_FLAGS_SETTING.update({
                    'FLAGS_cudnn_batchnorm_spatial_persistent': 1
                })
            paddle.set_flags(AMP_RELATED_FLAGS_SETTING)

            self.scale_loss = self.config["AMP"].get("scale_loss", 1.0)
            self.use_dynamic_loss_scaling = self.config["AMP"].get(
                "use_dynamic_loss_scaling", False)
            self.scaler = paddle.amp.GradScaler(
                init_loss_scaling=self.scale_loss,
                use_dynamic_loss_scaling=self.use_dynamic_loss_scaling)

            self.amp_level = self.config['AMP'].get("level", "O1")
            if self.amp_level not in ["O1", "O2"]:
                msg = "[Parameter Error]: The optimize level of AMP only support 'O1' and 'O2'. The level has been set 'O1'."
                logger.warning(msg)
                self.config['AMP']["level"] = "O1"
                self.amp_level = "O1"

            self.amp_eval = self.config["AMP"].get("use_fp16_test", False)
            # TODO(gaotingquan): Paddle not yet support FP32 evaluation when training with AMPO2
            if self.mode == "train" and self.config["Global"].get(
                    "eval_during_train",
                    True) and self.amp_level == "O2" and self.amp_eval == False:
                msg = "PaddlePaddle only support FP16 evaluation when training with AMP O2 now. "
                logger.warning(msg)
                self.config["AMP"]["use_fp16_test"] = True
                self.amp_eval = True

            # TODO(gaotingquan): to compatible with different versions of Paddle
            paddle_version = paddle.__version__[:3]
            # paddle version < 2.3.0 and not develop

            model, optimizer = paddle.amp.decorate(
                        models=self.model,
                        optimizers=self.optimizer,
                        level=self.amp_level,
                        save_dtype='float32')

            if (float(paddle_version) >=2.3 or float(paddle_version) == 0.0) and self.amp_level == "O2" and self.amp_eval:
                msg = "The PaddlePaddle that installed not support FP16 evaluation in AMP O2. Please use PaddlePaddle version >= 2.3.0. Use FP32 evaluation instead and please notice the Eval Dataset output_fp16 should be 'False'."
                logger.warning(msg)
                self.amp_eval = False
            else:
                self.model = model
                self.optimizer = optimizer

        # check the gpu num
        world_size = dist.get_world_size()
        self.config["Global"]["distributed"] = world_size != 1
        if self.mode == "train":
            std_gpu_num = 8 if isinstance(
                self.config["Optimizer"],
                dict) and self.config["Optimizer"]["name"] == "AdamW" else 4
            if world_size != std_gpu_num:
                msg = f"The training strategy provided by PaddleClas is based on {std_gpu_num} gpus. But the number of gpu is {world_size} in current training. Please modify the stategy (learning rate, batch size and so on) if use this config to train."
                logger.warning(msg)

    def train(self):
        assert self.mode == "train"
        print_batch_step = self.config['Global']['print_batch_step']
        save_interval = self.config["Global"]["save_interval"]
        best_metric = {
            "metric": -1.0,
            "epoch": 0,
        }
        ema_module = None
        if self.ema:
            best_metric_ema = 0.0
            ema_module = self.model_ema.module
        # key:
        # val: metrics list word
        self.output_info = dict()
        self.time_info = {
            "batch_cost": AverageMeter(
                "batch_cost", '.5f', postfix=" s,"),
            "reader_cost": AverageMeter(
                "reader_cost", ".5f", postfix=" s,"),
        }

        if self.config.Global.checkpoints is not None:
            metric_info = init_model(self.config.Global, self.model,
                                     self.optimizer, self.train_loss_func,
                                     ema_module)
            if metric_info is not None:
                best_metric.update(metric_info)

        for epoch_id in range(best_metric["epoch"] + 1,
                              self.config["Global"]["epochs"] + 1):
            acc = 0.0
            # for one epoch train
            self.train_epoch(epoch_id, print_batch_step)

            if self.use_dali:
                self.train_dataloader.reset()
            metric_msg = ", ".join(
                [self.output_info[key].avg_info for key in self.output_info])
            logger.info("[Train][Epoch {}/{}][Avg]{}".format(
                epoch_id, self.config["Global"]["epochs"], metric_msg))
            self.output_info.clear()

            # eval model and save model if possible
            start_eval_epoch = self.config["Global"].get("start_eval_epoch",
                                                         0) - 1
            if self.config["Global"][
                    "eval_during_train"] and epoch_id % self.config["Global"][
                        "eval_interval"] == 0 and epoch_id > start_eval_epoch:
                acc = self.eval(epoch_id)

                # step lr (by epoch) according to given metric, such as acc
                for i in range(len(self.lr_sch)):
                    if getattr(self.lr_sch[i], "by_epoch", False) and \
                            type_name(self.lr_sch[i]) == "ReduceOnPlateau":
                        self.lr_sch[i].step(acc)

                if acc > best_metric["metric"]:
                    best_metric["metric"] = acc
                    best_metric["epoch"] = epoch_id
                    save_load.save_model(
                        self.model,
                        self.optimizer,
                        best_metric,
                        self.output_dir,
                        ema=ema_module,
                        model_name=self.config["Arch"]["name"],
                        prefix="best_model",
                        loss=self.train_loss_func,
                        save_student_model=True)
                logger.info("[Eval][Epoch {}][best metric: {}]".format(
                    epoch_id, best_metric["metric"]))
                logger.scaler(
                    name="eval_acc",
                    value=acc,
                    step=epoch_id,
                    writer=self.vdl_writer)

                self.model.train()

                if self.ema:
                    ori_model, self.model = self.model, ema_module
                    acc_ema = self.eval(epoch_id)
                    self.model = ori_model
                    ema_module.eval()

                    if acc_ema > best_metric_ema:
                        best_metric_ema = acc_ema
                        save_load.save_model(
                            self.model,
                            self.optimizer,
                            {"metric": acc_ema,
                             "epoch": epoch_id},
                            self.output_dir,
                            ema=ema_module,
                            model_name=self.config["Arch"]["name"],
                            prefix="best_model_ema",
                            loss=self.train_loss_func)
                    logger.info("[Eval][Epoch {}][best metric ema: {}]".format(
                        epoch_id, best_metric_ema))
                    logger.scaler(
                        name="eval_acc_ema",
                        value=acc_ema,
                        step=epoch_id,
                        writer=self.vdl_writer)

            # save model
            if save_interval > 0 and epoch_id % save_interval == 0:
                save_load.save_model(
                    self.model,
                    self.optimizer, {"metric": acc,
                                     "epoch": epoch_id},
                    self.output_dir,
                    ema=ema_module,
                    model_name=self.config["Arch"]["name"],
                    prefix="epoch_{}".format(epoch_id),
                    loss=self.train_loss_func)
            # save the latest model
            save_load.save_model(
                self.model,
                self.optimizer, {"metric": acc,
                                 "epoch": epoch_id},
                self.output_dir,
                ema=ema_module,
                model_name=self.config["Arch"]["name"],
                prefix="latest",
                loss=self.train_loss_func)

        if self.vdl_writer is not None:
            self.vdl_writer.close()

    @paddle.no_grad()
    def eval(self, epoch_id=0):
        assert self.mode in ["train", "eval"]
        self.model.eval()
        eval_result = self.eval_epoch(epoch_id)
        self.model.train()
        return eval_result

    @paddle.no_grad()
    def infer(self):
        raise NotImplementedError

    def export(self):
        assert self.mode == "export"
        use_multilabel = self.config["Global"].get(
            "use_multilabel",
            False) or "ATTRMetric" in self.config["Metric"]["Eval"][0]
        model = ExportModel(self.config["Arch"], self.model, use_multilabel)
        if self.config["Global"]["pretrained_model"] is not None:
            if self.config["Global"]["pretrained_model"].startswith("http"):
                load_dygraph_pretrain_from_url(
                    model.base_model,
                    self.config["Global"]["pretrained_model"])
            else:
                load_dygraph_pretrain(
                    model.base_model,
                    self.config["Global"]["pretrained_model"])

        model.eval()

        # for rep nets
        for layer in self.model.sublayers():
            if hasattr(layer, "rep") and not getattr(layer, "is_repped"):
                layer.rep()

        save_path = os.path.join(self.config["Global"]["save_inference_dir"],
                                 "inference")

        model = paddle.jit.to_static(
            model,
            input_spec=[
                paddle.static.InputSpec(
                    shape=[None] + self.config["Global"]["image_shape"],
                    dtype='float32')
            ])
        if hasattr(model.base_model,
                   "quanter") and model.base_model.quanter is not None:
            model.base_model.quanter.save_quantized_model(model,
                                                          save_path + "_int8")
        else:
            paddle.jit.save(model, save_path)
        logger.info(
            f"Export succeeded! The inference model exported has been saved in \"{self.config['Global']['save_inference_dir']}\"."
        )
    
    @abstractmethod
    def train_epoch(self, epoch_id, print_batch_step):
        pass

    @abstractmethod
    def eval_epoch(self, epoch_id):
        pass


class ExportModel(TheseusLayer):
    """
    ExportModel: add softmax onto the model
    """

    def __init__(self, config, model, use_multilabel):
        super().__init__()
        self.base_model = model
        # we should choose a final model to export
        if isinstance(self.base_model, DistillationModel):
            self.infer_model_name = config["infer_model_name"]
        else:
            self.infer_model_name = None

        self.infer_output_key = config.get("infer_output_key", None)
        if self.infer_output_key == "features" and isinstance(self.base_model,
                                                              RecModel):
            self.base_model.head = IdentityHead()
        if use_multilabel:
            self.out_act = nn.Sigmoid()
        else:
            if config.get("infer_add_softmax", True):
                self.out_act = nn.Softmax(axis=-1)
            else:
                self.out_act = None

    def eval(self):
        self.training = False
        for layer in self.sublayers():
            layer.training = False
            layer.eval()

    def forward(self, x):
        x = self.base_model(x)
        if isinstance(x, list):
            x = x[0]
        if self.infer_model_name is not None:
            x = x[self.infer_model_name]
        if self.infer_output_key is not None:
            x = x[self.infer_output_key]
        if self.out_act is not None:
            if isinstance(x, dict):
                x = x["logits"]
            x = self.out_act(x)
        return x