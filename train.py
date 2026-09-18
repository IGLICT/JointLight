import torch, os, json
from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_relit import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser
from diffsynth.trainers.relit_dataset import UnifiedDataset
from diffsynth.pipelines.wan_video_relit import WanVideoUnit_EnvMapEmbedder
from diffsynth.models.wan_video_modality_identifier import WanModalityIdentifierModel


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None, model_json=None,
        resume_from_checkpoint=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        train_sa_only=False,
        tokenizer_path="models/Wan-AI/Wan2.2-TI2V-5B/google/umt5-xxl",
    ):
        super().__init__()
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, model_json, enable_fp8_training=False)
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs,
            tokenizer_config=ModelConfig(path=tokenizer_path), redirect_common_files=False,
        )


        self.pipe.dit.modality_identifier = WanModalityIdentifierModel()
        self.pipe.dit.modality_identifier.init()
        self.pipe.dit.modality_identifier.to(torch.bfloat16)

        if resume_from_checkpoint is not None:
            print(f"Loading from checkpoint: {resume_from_checkpoint}")
            state_dict = load_state_dict(resume_from_checkpoint)
            self.pipe.dit.load_state_dict(state_dict)

        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )
        
        if train_sa_only:
            self.freeze_parameters()
            for name, module in self.pipe.dit.named_modules():
                if any(keyword in name for keyword in [ "self_attn", "norm", "head", "gate", "modality_identifier"]):
                    print(f"Trainable: {name}")
                    for param in module.parameters():
                        param.requires_grad = True
        
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
    
    def freeze_parameters(self):
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.dit.train()
        
    def forward_preprocess(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        
        inputs_shared = {
            "input_video": data["video"],
            "cond_video": data["cond_video"],
            "shading_cue": data["shading_cue"],
            "shading_gt": data["shading_gt"],
            "match_shading": data["match_shading"],
            "shading_timestep_init_id": 1,
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }

        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    def envmap_preprocess(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        
        inputs_shared = {
            "envmap": data["envmap"],
            "env_rot": data["env_rot"],
            "cam_rot": data["cam_rot"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        
        inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(WanVideoUnit_EnvMapEmbedder(), self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}


    def forward(self, data, inputs=None):
        if inputs is None: 
            inputs = self.forward_preprocess(data)
        else:
            inputs.update(self.envmap_preprocess(inputs))
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    datasets = []
    for dataset_base_path, dataset_metadata_path, dataset_metadata_match_path, shading_cue_base_path, shading_gt_base_path, dataset_type, num_frames in zip(args.dataset_base_path, args.dataset_metadata_path, args.dataset_metadata_match_path, args.shading_cue_base_path, args.shading_gt_base_path, args.dataset_type, args.num_frames):
        main_data_operator = UnifiedDataset.default_video_operator(
            base_path=dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=32,
            width_division_factor=32,
            num_frames=num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        )
        dynamic_length_data_operator = UnifiedDataset.dynamic_length_video_operator(
            base_path=dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=32,
            width_division_factor=32,
            num_frames=num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        )
        shading_cue_data_operator = UnifiedDataset.dynamic_length_video_operator(
            base_path=shading_cue_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=32,
            width_division_factor=32,
            num_frames=num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        )
        shading_gt_data_operator = UnifiedDataset.dynamic_length_video_operator(
            base_path=shading_gt_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=32,
            width_division_factor=32,
            num_frames=num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        )
        datasets.append(
            UnifiedDataset(
                max_num_frames=num_frames,
                time_division_factor=4,
                time_division_remainder=1,
                base_path=dataset_base_path,
                shading_cue_base_path=shading_cue_base_path,
                shading_gt_base_path=shading_gt_base_path,
                metadata_path=dataset_metadata_path,
                metadata_match_path=dataset_metadata_match_path,
                hdri_base_path=args.hdri_base_path,
                repeat=args.dataset_repeat,
                data_file_keys=args.data_file_keys.split(","),
                load_from_cache=args.load_from_cache,
                dynamic_length=args.dynamic_length,
                uniform_input=args.uniform_input,
                main_data_operator=main_data_operator,
                dynamic_length_data_operator=dynamic_length_data_operator,
                shading_cue_data_operator=shading_cue_data_operator,
                shading_gt_data_operator=shading_gt_data_operator,
                dataset_type=dataset_type,
                shading_cue_as_input=args.shading_cue_as_input if shading_cue_base_path != "" else False,
                start_match_shading=args.start_match_shading,
                end_match_shading=args.end_match_shading
            )
        )
    dataset = torch.utils.data.ConcatDataset(datasets)

    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        model_json=args.model_json,
        resume_from_checkpoint=args.resume_from_checkpoint,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        train_sa_only=args.train_sa_only,
        tokenizer_path=args.tokenizer_path,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )
    launch_training_task(dataset, model, model_logger, args=args)
