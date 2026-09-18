import torch
from diffsynth.data.video import VideoData, save_video
from diffsynth.models.utils import load_state_dict
from diffsynth.pipelines.wan_video_relit import WanVideoPipeline, ModelConfig
from diffsynth.models.wan_video_modality_identifier import WanModalityIdentifierModel
import os
import json
import argparse
import itertools
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Video Relighting with Shading Conditioning")

    parser.add_argument("--model_config", type=str, default="model_configs.json", help="Path to model config JSON")
    parser.add_argument("--checkpoint", type=str, default="models/JointLight/JointLight.safetensors", help="Path to a JointLight safetensors checkpoint")
    parser.add_argument("--tokenizer_path", default="models/Wan-AI/Wan2.2-TI2V-5B/google/umt5-xxl", help="Local UMT5 tokenizer directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    parser.add_argument("--video_path", type=str, required=True, help="Cond video path")

    parser.add_argument("--input_video_path", type=str, required=False, default=None, help="Input video path")

    parser.add_argument("--shading_cue_paths", type=str, nargs='+', required=True, help="List of shading cue video paths")


    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames to generate")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="Classifier-free guidance scale")
    parser.add_argument("--shading_timestep_init", type=float, default=1.0, help="init timestep for input shading")
    parser.add_argument("--input_timestep_init", type=float, default=1.0, help="init timestep for cond video")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--height", type=int, default=832, help="Output video height")
    parser.add_argument("--width", type=int, default=480, help="Output video width")

    parser.add_argument("--output_dir", type=str, default="results", help="Base output directory")

    args = parser.parse_args()

    if args.num_frames < 1 or args.num_frames % 4 != 1:
        parser.error("--num_frames must be positive and equal to 4n+1 (e.g. 1, 49, 81)")
    if args.height <= 0 or args.width <= 0 or args.height % 32 or args.width % 32:
        parser.error("--height and --width must be positive multiples of 32")
    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.is_file():
        parser.error(f"Checkpoint does not exist: {checkpoint_path}")


    with open(args.model_config, "r", encoding="utf-8") as f:
        model_paths = json.load(f)
        model_configs = [ModelConfig(path=path, offload_device='cpu') for path in model_paths]

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=model_configs,
        tokenizer_config=ModelConfig(path=args.tokenizer_path),
        redirect_common_files=False,
    )


    pipe.dit.modality_identifier = WanModalityIdentifierModel()
    pipe.dit.modality_identifier.init()
    pipe.dit.modality_identifier.to(torch.bfloat16)

    state_dict = load_state_dict(str(checkpoint_path))
    pipe.dit.load_state_dict(state_dict, strict=True)
    del state_dict
    pipe.to(args.device)

    cond_video = VideoData(args.video_path, height=args.height, width=args.width).raw_data()[:args.num_frames]
    input_video = VideoData(args.input_video_path, height=args.height, width=args.width).raw_data()[:args.num_frames] if args.input_video_path is not None else None

    if len(args.shading_cue_paths) == 1 and os.path.isdir(args.shading_cue_paths[0]):
        shading_cue_names = sorted(itertools.chain(*(Path(args.shading_cue_paths[0]).rglob(f'*.{suffix}') for suffix in ['mp4'])))
        shading_cue_names = [str(name.relative_to(args.shading_cue_paths[0])) for name in shading_cue_names]
        shading_cue_paths = [os.path.join(args.shading_cue_paths[0], name) for name in shading_cue_names]
    else:
        shading_cue_paths = args.shading_cue_paths
    shading_cues = [
        VideoData(path, height=args.height, width=args.width).raw_data()[:args.num_frames]
        for path in shading_cue_paths
    ]
    if not shading_cues:
        raise ValueError("No shading cue videos found")
    for name, frames in [(args.video_path, cond_video), *zip(shading_cue_paths, shading_cues)]:
        if len(frames) != args.num_frames:
            raise ValueError(f"{name}: expected at least {args.num_frames} frames, got {len(frames)}")
    if input_video is not None and len(input_video) != args.num_frames:
        raise ValueError(f"Input video must contain at least {args.num_frames} frames")

    video_name = '_'.join(args.video_path.split('/')[-3:])
    save_dir = os.path.join(
        args.output_dir,
        checkpoint_path.stem,
        f"frames_{args.num_frames}",
        f"cfg_{args.cfg_scale}",
        f"shading_timestep_init_{args.shading_timestep_init}",
        f"input_timestep_init_{args.input_timestep_init}",
        f"hw_{args.height}_{args.width}",
        video_name
    )
    os.makedirs(save_dir, exist_ok=True)

    for i, shading_cue in enumerate(shading_cues):
        video, shading_video = pipe(
            prompt=args.prompt,
            negative_prompt="",
            cond_video=cond_video,
            input_video=input_video if input_video is not None else cond_video,
            shading_gt=shading_cue,
            shading_timestep_init=args.shading_timestep_init,
            input_timestep_init=args.input_timestep_init,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            cfg_scale=args.cfg_scale,
            seed=args.seed,
            tiled=False,
        )
        output_name = '_'.join(shading_cue_paths[i].split('/')[-3:])
        output_shading_name = output_name.replace('.mp4', '_shading.mp4')
        save_path = os.path.join(save_dir, output_name)
        save_shading_path = os.path.join(save_dir, output_shading_name)
        save_video(video, save_path, fps=30, quality=5)
        print(f"✅ Saved video to {save_path}")
        save_video(shading_video, save_shading_path, fps=30, quality=5)
        print(f"✅ Saved shading video to {save_shading_path}")

if __name__ == "__main__":
    main()
