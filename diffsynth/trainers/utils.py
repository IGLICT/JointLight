import os, torch, warnings, argparse, json, ast
from ..utils import ModelConfig
from ..models.utils import load_state_dict
from peft import LoraConfig, inject_adapter_in_model
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import WeightedRandomSampler

class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()


    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self


    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules


    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names


    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None, upcast_dtype=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        if upcast_dtype is not None:
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.to(upcast_dtype)
        return model


    def mapping_lora_state_dict(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if "lora_A.weight" in key or "lora_B.weight" in key:
                new_key = key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
                new_state_dict[new_key] = value
            elif "lora_A.default.weight" in key or "lora_B.default.weight" in key:
                new_state_dict[key] = value
        return new_state_dict


    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict


    def transfer_data_to_device(self, data, device):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
        return data


    def parse_model_configs(self, model_paths, model_id_with_origin_paths, model_json, enable_fp8_training=False):
        offload_dtype = torch.float8_e4m3fn if enable_fp8_training else None
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path, offload_dtype=offload_dtype) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1], offload_dtype=offload_dtype) for i in model_id_with_origin_paths]
        if model_json is not None:
            with open(model_json, "r", encoding="utf-8") as f:
                model_paths = json.load(f) 
                model_configs += [ModelConfig(path=path, offload_dtype=offload_dtype) for path in model_paths]
        return model_configs


    def switch_pipe_to_training_mode(
        self,
        pipe,
        trainable_models,
        lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=None,
        enable_fp8_training=False,
    ):
        # Scheduler
        pipe.scheduler.set_timesteps(1000, training=True)
        
        # Freeze untrainable models
        pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        
        # Enable FP8 if pipeline supports
        if enable_fp8_training and hasattr(pipe, "_enable_fp8_lora_training"):
            pipe._enable_fp8_lora_training(torch.float8_e4m3fn)
        
        # Add LoRA to the base models
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank,
                upcast_dtype=pipe.torch_dtype,
            )
            if lora_checkpoint is not None:
                state_dict = load_state_dict(lora_checkpoint)
                state_dict = self.mapping_lora_state_dict(state_dict)
                load_result = model.load_state_dict(state_dict, strict=False)
                print(f"LoRA checkpoint loaded: {lora_checkpoint}, total {len(state_dict)} keys")
                if len(load_result[1]) > 0:
                    print(f"Warning, LoRA key mismatch! Unexpected keys in LoRA checkpoint: {load_result[1]}")
            setattr(pipe, lora_base_model, model)


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0
        self.num_epochs = 0


    def on_step_end(self, accelerator, model, save_steps=None):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"epoch-{self.num_epochs}-step-{self.num_steps}.safetensors")


    def on_epoch_end(self, accelerator, model, epoch_id):
        self.num_epochs += 1
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


    def on_training_end(self, accelerator, model, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def save_model(self, accelerator, model, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)


def launch_training_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 8,
    save_steps: int = None,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    find_unused_parameters: bool = False,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        gradient_accumulation_steps = args.gradient_accumulation_steps
        find_unused_parameters = args.find_unused_parameters
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    ratios = args.dataset_ratio  # a:b:c:d = 8:1:3:2
    weights = []
    for ds, ratio in zip(dataset.datasets, ratios):
        weight_per_sample = ratio / len(ds)
        weights.extend([weight_per_sample] * len(ds))
    num_samples = len(dataset)  # 或设为更大值以增加 epoch 长度
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples,
        replacement=True  # 允许重复采样（推荐）
    )
    # dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        collate_fn=lambda x: x[0],
        # shuffle=False,  # sampler 和 shuffle 互斥
        num_workers=num_workers
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    step = 0
    for epoch_id in range(num_epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_id}/{num_epochs}")
        for data in progress_bar:
            try:
                with accelerator.accumulate(model):
                    optimizer.zero_grad()
                    if data["load_from_cache"]:
                        data['noise'] = torch.randn_like(data['input_latents'])   # b c f h w
                        loss_dict = model({}, inputs=data)
                    else:
                        loss_dict = model(data)
                    rgb_loss = loss_dict['rgb_loss']
                    shading_loss = loss_dict['shading_loss']
                    # loss = (rgb_loss + 0.1 * shading_loss) * loss_dict['scheduler_weight']
                    # loss = (rgb_loss + shading_loss) * loss_dict['scheduler_weight']
                    loss = rgb_loss * loss_dict['scheduler_weight_rgb'] + shading_loss * loss_dict['scheduler_weight_shading']
                    accelerator.backward(loss)
                    optimizer.step()
                    model_logger.on_step_end(accelerator, model, save_steps)
                    scheduler.step()
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'rgb_loss': f'{rgb_loss.item():.4f}',
                        'shading_loss': f'{shading_loss.item():.4f}',
                        'type': f"{data['data_type']}"
                    })
                    # progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                    # progress_bar.set_postfix({'type': f"{data['data_type']}"})
                    step += 1
                    for d in dataset.datasets: d.set_step(step)
            except Exception as e:
                print(f"Error processing data: {e}")
                import traceback
                traceback.print_exc()
                continue
        model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)


def parse_path_list(s):
    """
    Parse a string into a list of paths.
    Supports:
      - Single path: "xxx" → ["xxx"]
      - JSON list: '["xxx", "yyy"]' → ["xxx", "yyy"]
      - Python list: "['xxx', 'yyy']" → ["xxx", "yyy"]
    """
    s = s.strip()
    if not s:
        return []
    
    # Case 1: Looks like a list (starts with [ and ends with ])
    if s.startswith('[') and s.endswith(']'):
        try:
            # Try JSON first (more strict)
            return json.loads(s)
        except json.JSONDecodeError:
            try:
                # Fall back to ast.literal_eval (handles single quotes)
                result = ast.literal_eval(s)
                if isinstance(result, list):
                    return result
                else:
                    raise ValueError("Not a list")
            except (ValueError, SyntaxError):
                raise argparse.ArgumentTypeError(f"Invalid list format: {s}")
    else:
        # Case 2: Single path
        return [s]


def parse_int_list(s):
    """Parse comma-separated integers like '256,512,1024' into [256, 512, 1024]"""
    if s is None or s.strip() == '':
        return None
    try:
        return [int(x.strip()) for x in s.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid height list: {s}. Expected comma-separated integers like '256,512'.")


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--tokenizer_path", default="models/Wan-AI/Wan2.2-TI2V-5B/google/umt5-xxl", help="Local UMT5 tokenizer directory")
    parser.add_argument(
        "--dataset_base_path",
        type=parse_path_list,
        required=True,
        help="Base path(s) of the dataset. Can be a single path or a JSON/Python list of paths."
    )
    parser.add_argument(
        "--dataset_metadata_path",
        type=parse_path_list,
        required=True,
        help="Path to the metadata file of the dataset."
    )
    parser.add_argument(
        "--dataset_metadata_match_path",
        type=parse_path_list,
        required=True,
        help="Path to the metadata_match file of the dataset."
    )
    parser.add_argument(
        "--shading_cue_base_path",
        type=parse_path_list,
        required=True,
        help="Base path(s) of the shading cue of the dataset. Can be a single path or a JSON/Python list of paths."
    )
    parser.add_argument(
        "--shading_gt_base_path",
        type=parse_path_list,
        required=True,
        help="Base path(s) of the shading gt of the dataset. Can be a single path or a JSON/Python list of paths."
    )
    parser.add_argument(
        "--dataset_type",
        type=parse_int_list,
        required=True,
        help="list of dataset type. 0 for paired image dataset with hdris, 1 for paired image dataset without hdris, 2 for unpaired image datset without hdris, 3 for paired video dataset with hdris, 4 for paired video dataset without hdris, 5 for unpaired video datset without hdris, 6 for real video dataset with anotated shading and degradation pairs"
    )
    parser.add_argument(
        "--shading_cue_as_input",
        type=bool,
        required=False,
        default=False,
        help="whether to use shading cue as input to match shading gap"
    )
    parser.add_argument(
        "--train_sa_only",
        type=bool,
        required=False,
        default=False,
        help="whether to freeze cross attention"
    )
    parser.add_argument(
        "--start_match_shading",
        type=int,
        required=False,
        default=0,
        help="list of dataset type. 0 for paired image dataset with hdris, 1 for paired image dataset without hdris, 2 for unpaired image datset without hdris, 3 for paired video dataset with hdris, 4 for paired video dataset without hdris, 5 for unpaired video datset without hdris, 6 for real video dataset with anotated shading and degradation pairs"
    )
    parser.add_argument(
        "--end_match_shading",
        type=int,
        required=False,
        default=3000,
        help="list of dataset type. 0 for paired image dataset with hdris, 1 for paired image dataset without hdris, 2 for unpaired image datset without hdris, 3 for paired video dataset with hdris, 4 for paired video dataset without hdris, 5 for unpaired video datset without hdris, 6 for real video dataset with anotated shading and degradation pairs"
    )
    parser.add_argument(
        "--dataset_ratio",
        type=parse_int_list,
        required=True,
        help="list of dataset ratios. e.g. 1,1 for 1:1"
    )
    parser.add_argument(
        "--num_frames",
        type=parse_int_list,
        required=True,
        help="Number of frames per video for each dataset. Frames are sampled from the video prefix."
    )
    # parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    # parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--hdri_base_path", type=str, default="", required=True, help="Base path of the hdri dataset.")
    parser.add_argument("--max_pixels", type=int, default=1280*720, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    # parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video. Frames are sampled from the video prefix.")
    parser.add_argument("--data_file_keys", type=str, default="image,video", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_json", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--load_from_cache", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--dynamic_length", default=False, action="store_true", help="Whether to use dynamic frame length.")
    parser.add_argument("--uniform_input", default=False, action="store_true", help="Whether to use uniform lighting videos as input.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path of checkpoint to resume from.")
    
    return parser
