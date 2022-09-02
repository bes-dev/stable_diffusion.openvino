import numpy as np
# openvino
from openvino.runtime import Core
# onnxruntime
from onnxruntime import InferenceSession
# models hub
from huggingface_hub import hf_hub_download


class CoreOV(object):
    __instance = None

    def __new__(cls, *args, **kwargs):
        if not CoreOV.__instance:
            CoreOV.__instance = object.__new__(cls)
        return CoreOV.__instance

    def __init__(self):
        self.core = Core()


class EngineOV:
    def __init__(self, repo_id, model_id, device="CPU"):
        self._core = CoreOV()
        self._model = self._core.core.read_model(
            hf_hub_download(repo_id=repo_id, filename=f"{model_id}.xml"),
            hf_hub_download(repo_id=repo_id, filename=f"{model_id}.bin")
        )
        self._model_ex = self._core.core.compile_model(self._model, device)

    def _result(self, var):
        return next(iter(var.values()))

    def __call__(self, args):
        return self._result(self._model_ex.infer_new_request(args))

    def inputs(self):
        return self._model.inputs


class EngineONNX:
    def __init__(self, repo_id, model_id, device="CPU"):
        self.model = InferenceSession(hf_hub_download(repo_id=repo_id, filename=f"{model_id}.onnx"))

    def __call__(self, args):
        return self.model.run(None, args)[0]

    def inputs(self):
        return self.model.get_inputs()
