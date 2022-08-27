import collections

import numpy as np
import torch
import onnxruntime as onr
from utils.alarm import show_warning


class ProductionModel:
    """
    Model Inference Wrapper.
    """

    def __init__(self,
                 model_path: str,
                 model_type: str,
                 session_provider: str = None,
                 output_tasks: int = 1,
                 compatible_mode: bool = True,
                 model_container: torch.nn.Module = None):
        """
        Initialize the model wrapper

        :param model_path: Path of the model checkpoint
        :param model_type: Type of the checkpoint ["pytorch","onnx"]
        :param session_provider: ONNXRuntime session provider ["cpu","cuda","tensor_rt"]
        """
        self.model_type = model_type
        self.session_provider = session_provider
        self.output_tasks = output_tasks
        self.compat = compatible_mode
        if compatible_mode and output_tasks > 1:
            raise Exception("Cannot Handle Multitask Output (in compat mode)")
        if model_type == "pytorch":
            self.model_p = torch.load(model_path)

            if type(self.model_p) is collections.OrderedDict:
                self.model = model_container
                self.model.load_state_dict(self.model_p)
            else:
                self.model = self.model_p
            self.model = self.model.to("cpu")
            show_warning("Deploy the runner to get faster speed during inference! Pytorch inference"
                         "is deprecated. To deploy the runner, run deploy.py")
        elif model_type == "onnx":
            provider = None
            if session_provider == "cpu":
                provider = ["CPUExecutionProvider"]
            elif session_provider == "tensor_rt":
                provider = ["TensorrtExecutionProvider"]
            elif session_provider == "cuda":
                provider = ["CUDAExecutionProvider"]
            else:
                raise Exception("Session provider for ONNXRuntime is not supported")
            print("Provider", provider)
            self.model = onr.InferenceSession(model_path, providers=provider)
            self.input_name = self.model.get_inputs()[0].name
            self.output_names_ls = self.model.get_outputs()
            self.output_names = []
            for i in self.output_names_ls:
                self.output_names.append(i.name)
        else:
            self.model = None
            raise Exception("Inference runner is not supported. Supported types:[pytorch/onnx]")

    def infer(self, src):
        """
        Output the result of the model with given input

        :param src: Input data
        :return: Output of the model
        """
        if self.model_type == "pytorch":
            src = np.expand_dims(src, 0)
            if type(src) is np.ndarray:
                src = torch.tensor(src)
            if self.compat:
                ret = self.model(src).detach().numpy()
                return ret[0]
            else:
                res = self.model(src)
                ret = tuple([res[i].detach().numpy()[0] for i in range(self.output_tasks)])
                return ret

        elif self.model_type == "onnx":
            src = np.expand_dims(src, 0)
            input_feed = {self.input_name: src}
            res = self.model.run(self.output_names, input_feed=input_feed)
            if self.compat:
                return res[0][0]
            else:
                ret = tuple([res[i][0] for i in range(self.output_tasks)])
                return ret
