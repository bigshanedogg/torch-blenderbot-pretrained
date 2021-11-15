from typing import List
from torch import nn
from tqdm import tqdm
from transformer.data.blender_bot_data_loader import RetrieverEncoderDataLoader, RetrieverFinetuningDataLoader, GeneratorPretrainingDataLoader, GeneratorFinetuningDataLoader
from transformer.trainers.interface import TrainerInterface
from transformer.trainers.bert_trainer import BertTrainer
from transformer.trainers.transformer_trainer import TransformerTrainer
from transformer.trainers.gpt_trainer import GptTrainer
from transformer.trainers.poly_encoder_trainer import PolyEncoderTrainer
from transformer.trainers.utils import save_dialog_history_set, save_dialog_response_set, load_encoded_context_set, load_encoded_candidate_set

class RetrieverEncoderBertTrainer(BertTrainer):
    def create_data_loader(self, **params):
        data_loader = RetrieverEncoderDataLoader(**params)
        return data_loader

class RetrieverFinetuningPolyEncoderTrainer(PolyEncoderTrainer):
    def create_data_loader(self, **params):
        data_loader = RetrieverFinetuningDataLoader(**params)
        return data_loader

    def get_data_loader_params(self, **kwargs):
        common_required_parameters = TrainerInterface.get_data_loader_params(self=self, **kwargs)
        custom_required_parameters = ["timesteps", "embedding_dict", "left_sep_tokens", "left_fixed_segment_id", "right_sep_tokens", "right_fixed_segment_id"]
        self.assert_contain_elements(required=custom_required_parameters, target=kwargs, name="custom_required_parameters")
        required_parameters = common_required_parameters + custom_required_parameters
        default_parameters = [("num_workers", self.num_workers), ("pin_memory", self.pin_memory), ("additional_responses", []), ("approach", "ignore")]
        params = dict()
        for k, v in default_parameters:
            if k in kwargs: v = kwargs.pop(k)
            params[k] = v
        for k, v in kwargs.items():
            if k not in required_parameters: continue
            params[k] = v
        return params

    def save_additional(self, **kwargs):
        required_parameters = ["path", "model", "train_data_loader", "device"]
        self.assert_contain_elements(required=required_parameters, target=kwargs, name="parameters")
        path = kwargs.pop("path")
        model = kwargs.pop("model")
        train_data_loader = kwargs.pop("train_data_loader")
        device = kwargs.pop("device")

        self.save_encoded_candidate_set(path=path, model=model, data_loader=train_data_loader, device=device)
        candidates, encoded_candidates = self.load_encoded_candidate_set(path=path)
        self.save_encoded_context_set(path=path, model=model, data_loader=train_data_loader, device=device, encoded_candidates=encoded_candidates)
        return path

    def save_encoded_context_set(self, path: str, model:nn.modules.Module, data_loader: RetrieverFinetuningDataLoader, device, encoded_candidates: List[List[int]] = None):
        if encoded_candidates is None:
            candidates, encoded_candidates = self.extract_encoded_candidate_set(model=model, data_loader=data_loader, device=device)
            dialog_response_set = candidates, encoded_candidates
            save_dialog_response_set(path=path, data=dialog_response_set)
            
        contexts, encoded_contexts = self.extract_encoded_context_set(model=model, data_loader=data_loader, device=device, encoded_candidates=encoded_candidates)
        dialog_history_set = contexts, encoded_contexts
        path = save_dialog_history_set(path=path, data=dialog_history_set)
        return path

    def save_encoded_candidate_set(self, path: str, model:nn.modules.Module, data_loader: RetrieverFinetuningDataLoader, device):
        candidates, encoded_candidates = self.extract_encoded_candidate_set(model=model, data_loader=data_loader, device=device)
        dialog_response_set = candidates, encoded_candidates
        path = save_dialog_response_set(path=path, data=dialog_response_set)
        return path

    def load_encoded_context_set(self, path: str):
        contexts, encoded_contexts = load_encoded_context_set(path=path)
        return contexts, encoded_contexts

    def load_encoded_candidate_set(self, path: str):
        candidates, encoded_candidates = load_encoded_candidate_set(path=path)
        return candidates, encoded_candidates

    def extract_encoded_context_set(self, model:nn.modules.Module, data_loader: RetrieverFinetuningDataLoader, device, encoded_candidates):
        self.assert_isintance(obj=model, data_type=nn.modules.Module)
        self.assert_equal(a=model.__name__, b="poly_encoder")
        self.assert_isintance(obj=data_loader, data_type=RetrieverFinetuningDataLoader)
        model = TrainerInterface.set_device(obj=model, device=device)

        if encoded_candidates is None:
            candidates, encoded_candidates = self.extract_encoded_candidate_set(model=model, data_loader=data_loader, device=device)
        candidate_embed = self.convert_to_tensor(data=encoded_candidates, device=device)

        contexts = []
        encoded_contexts = []
        extract_verbose = self.verbose_template[:-1].format(mode="Extract_context", device=device, idx=1, num_iters=1)
        dataset_iter = tqdm(enumerate(data_loader.dataset), initial=0, total=len(data_loader.dataset), desc=extract_verbose, bar_format="{l_bar}{r_bar}")
        for row_idx, row in dataset_iter:
            context_input_row, candidate_input_row = data_loader.parse_row(row=row)
            context = context_input_row["token"][0][0]
            if context in contexts: continue

            batch = data_loader.preprocessor.encode(left_inputs=[context_input_row], right_inputs=[candidate_input_row], timesteps=data_loader.timesteps,
                                                    left_sep_tokens=data_loader.left_sep_tokens, right_sep_tokens=data_loader.right_sep_tokens,
                                                    left_fixed_segment_id=data_loader.left_fixed_segment_id, right_fixed_segment_id=data_loader.right_fixed_segment_id, approach=data_loader.approach)
            batch = [{k: self.convert_to_tensor(data=v, device=device) for k, v in _batch.items()} for _batch in batch]
            context_inputs, candidate_inputs, outputs = batch
            if len(candidate_inputs["token"]) <= 0: continue

            context_output = model.forward_context_encoder(context_inputs=context_inputs, candidate_embed=candidate_embed)[0]
            encoded_context = context_output["context_embed"]
            encoded_context = self.convert_to_numpy(tensor=encoded_context)
            encoded_context = encoded_context.tolist()
            contexts.append(context)
            encoded_contexts.append(encoded_context)
        return contexts, encoded_contexts

    def extract_encoded_candidate_set(self, model:nn.modules.Module, data_loader: RetrieverFinetuningDataLoader, device):
        self.assert_isintance(obj=model, data_type=nn.modules.Module)
        self.assert_equal(a=model.__name__, b="poly_encoder")
        self.assert_isintance(obj=data_loader, data_type=RetrieverFinetuningDataLoader)
        model = TrainerInterface.set_device(obj=model, device=device)

        candidates = []
        encoded_candidates = []
        extract_verbose = self.verbose_template[:-1].format(mode="Extract_candidate", device=device, idx=1, num_iters=1)
        dataset_iter = tqdm(enumerate(data_loader.dataset), initial=0, total=len(data_loader.dataset), desc=extract_verbose, bar_format="{l_bar}{r_bar}")
        for row_idx, row in dataset_iter:
            context_input_row, candidate_input_row = data_loader.parse_row(row=row)
            candidate = candidate_input_row["token"][0][0]
            if candidate in candidates: continue

            batch = data_loader.preprocessor.encode(left_inputs=[context_input_row], right_inputs=[candidate_input_row], timesteps=data_loader.timesteps,
                                                    left_sep_tokens=data_loader.left_sep_tokens, right_sep_tokens=data_loader.right_sep_tokens,
                                                    left_fixed_segment_id=data_loader.left_fixed_segment_id, right_fixed_segment_id=data_loader.right_fixed_segment_id, approach=data_loader.approach)
            batch = [{k: self.convert_to_tensor(data=v, device=device) for k, v in _batch.items()} for _batch in batch]
            context_inputs, candidate_inputs, outputs = batch
            if len(candidate_inputs["token"]) <= 0: continue

            candidate_output = model.forward_candidate_encoder(candidate_inputs=candidate_inputs)[0]
            encoded_candidate = candidate_output["candidate_embed"]
            encoded_candidate = self.convert_to_numpy(tensor=encoded_candidate)
            encoded_candidate = encoded_candidate.tolist()
            candidates.append(candidate)
            encoded_candidates.append(encoded_candidate)
        return candidates, encoded_candidates

class GeneratorPretrainingTransformerTrainer(TransformerTrainer):
    def create_data_loader(self, **params):
        data_loader = GeneratorPretrainingDataLoader(**params)
        return data_loader

class GeneratorFinetuningTransformerTrainer(TransformerTrainer):
    def create_data_loader(self, **params):
        data_loader = GeneratorFinetuningDataLoader(**params)
        return data_loader

    def get_data_loader_params(self, **kwargs):
        common_required_parameters = TrainerInterface.get_data_loader_params(self=self, **kwargs)
        custom_required_parameters = ["src_timesteps", "tgt_timesteps", "embedding_dict", "src_sep_tokens", "alpha", "use_condition"]
        self.assert_contain_elements(required=custom_required_parameters, target=kwargs, name="custom_required_parameters")
        required_parameters = common_required_parameters + custom_required_parameters
        default_parameters = [("num_workers", self.num_workers), ("pin_memory", self.pin_memory), ("approach", "ignore")]
        params = dict()
        for k, v in default_parameters:
            if k in kwargs: v = kwargs.pop(k)
            params[k] = v
        for k, v in kwargs.items():
            if k not in required_parameters: continue
            params[k] = v
        return params

class GeneratorFinetuningGptTrainer(GptTrainer):
    def create_data_loader(self, **params):
        data_loader = GeneratorFinetuningDataLoader(**params)
        return data_loader

    def get_data_loader_params(self, **kwargs):
        common_required_parameters = TrainerInterface.get_data_loader_params(self=self, **kwargs)
        custom_required_parameters = ["timesteps", "embedding_dict", "src_sep_tokens", "use_condition", "alpha"]
        self.assert_contain_elements(required=custom_required_parameters, target=kwargs, name="custom_required_parameters")
        required_parameters = common_required_parameters + custom_required_parameters
        default_parameters = [("num_workers", self.num_workers), ("pin_memory", self.pin_memory), ("approach", "ignore")]
        params = dict()
        for k, v in default_parameters:
            if k in kwargs: v = kwargs.pop(k)
            params[k] = v
        for k, v in kwargs.items():
            if k == "timesteps":
                params["src_timesteps"] = v
                params["tgt_timesteps"] = v
                continue
            if k not in required_parameters: continue
            params[k] = v
        return params