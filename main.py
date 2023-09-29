import argparse
import glob
import os
from pathlib import Path
import uuid
from src.pipelines.pipeline_animatediff_pix2pix import StableDiffusionInstructPix2PixPipeline
from diffusers import EulerAncestralDiscreteScheduler
import torch
from src.models.unet import UNet3DConditionModel
import numpy as np
from PIL import Image
import imageio

def convert_frames_to_mp4(frames, filename, fps=30):
    """Converts a list of PIL Image frames to an MP4 file.

    Args:
        frames: A list of PIL Image frames.
        filename: The name of the MP4 file to save.
        fps: Frames per second for the video.

    Returns:
        None
    """
    # Convert PIL Images to numpy arrays
    numpy_frames = [np.array(frame) for frame in frames]
    # Write frames to mp4
    imageio.mimwrite(filename, numpy_frames, fps=fps)

def convert_frames_to_gif(frames, filename, duration=100):
    """Converts a list of PIL Image frames to a GIF file.

    Args:
        frames: A list of PIL Image frames.
        filename: The name of the GIF file to save.
        duration: Duration of each frame in milliseconds.

    Returns:
        None
    """
    frames[0].save(
        filename, 
        save_all=True, 
        append_images=frames[1:], 
        loop=0, 
        duration=duration
    )


def convert_frames_to_gif_with_fps(frames, filename, fps=30):
    """Converts a list of PIL Image frames to a GIF file using fps.

    Args:
        frames: A list of PIL Image frames.
        filename: The name of the GIF file to save.
        fps: Frames per second for the gif.

    Returns:
        None
    """
    duration = 1000 // fps
    frames[0].save(
        filename, 
        save_all=True, 
        append_images=frames[1:], 
        loop=0, 
        duration=duration
    )


def run(t2i_model,
        prompt="", 
        negative_prompt="", 
        frame_count=16,
        num_inference_steps=20,
        guidance_scale=7.5,
        image_guidance_scale=1.5,
        width=512,
        height=512,
        dtype="float16",
        output_frames_directory="output_frames",
        output_video_directory="output_video",
        output_gif_directory="output_gif",
        motion_module="viddle/viddle-pix2pix-animatediff-v1.ckpt", 
        init_image=None,
        init_folder=None, 
        seed=42,
        fps=15,
        no_save_frames=False,
        no_save_video=False,
        no_save_gif=False,
        ):
  scheduler_kwargs = {
    "num_train_timesteps": 1000,
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "beta_schedule": "linear",
  }

  device = "cuda" if torch.cuda.is_available() else "cpu"
  if dtype == "float16":
     dtype = torch.float16
     variant = "fp16"
  elif dtype == "float32":
      dtype = torch.float32
      variant = "fp32"

  unet_additional_kwargs = {
    "in_channels": 8,
    "unet_use_cross_frame_attention": False,
    "unet_use_temporal_attention": False,
    "use_motion_module": True,
    "motion_module_resolutions": [1, 2, 4, 8],
    "motion_module_mid_block": False,
    "motion_module_decoder_only": False,
    "motion_module_type": "Vanilla",
    "motion_module_kwargs": {
        "num_attention_heads": 8,
        "num_transformer_block": 1,
        "attention_block_types": ["Temporal_Self", "Temporal_Self"],
        "temporal_position_encoding": True,
        "temporal_position_encoding_max_len": 32,
        "temporal_attention_dim_div": 1,
    },
  }

  pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
      t2i_model,
      scheduler=EulerAncestralDiscreteScheduler(**scheduler_kwargs),
      safety_checker=None,
      feature_extractor=None,
      requires_safety_checker=False,
      torch_dtype=dtype,
      variant=variant,
  ).to(device)

  pipeline.unet = UNet3DConditionModel.from_pretrained_unet(pipeline.unet,
                                                            unet_additional_kwargs=unet_additional_kwargs,
                                                            ).to(device=device, dtype=dtype)
  
  pipeline.enable_vae_slicing()

  motion_module_state_dict = torch.load(motion_module, map_location="cpu")
  _, unexpected = pipeline.unet.load_state_dict(motion_module_state_dict, strict=False)
  assert len(unexpected) == 0

  if init_image is not None and init_folder is None:
    image = Image.open(init_image)
    image = image.resize((width, height))
  elif init_folder is not None and init_image is None:
    image_paths = glob.glob(init_folder + "/*.png")
    # add the jpgs
    image_paths += glob.glob(init_folder + "/*.jpg")
    image_paths.sort()
    image_paths = image_paths[:frame_count]
  
    image = []

    for image_path in image_paths:
      image.append(Image.open(image_path).resize((width, height)))
  else:
    raise ValueError("Must provide either init_image or init_folder but not both")
  
  generator = torch.Generator(device=device).manual_seed(seed)

  frames = pipeline(prompt=prompt, 
                negative_prompt=negative_prompt, 
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                image_guidance_scale=image_guidance_scale,
                image=image,
                video_length=frame_count,
                generator=generator,
                )[0]

  # create a uuid prefix for the output files
  uuid_prefix = str(uuid.uuid4())

  if not no_save_frames:
    # Create output directory
    Path(output_frames_directory).mkdir(parents=True, exist_ok=True)

    # make the specific directory for this run
    output_frames_directory = os.path.join(output_frames_directory, uuid_prefix)
    Path(output_frames_directory).mkdir(parents=True, exist_ok=True)
    # Save frames
    for i, frame in enumerate(frames):
      frame.save(os.path.join(output_frames_directory, f"{str(i).zfill(4)}.png"))

  if not no_save_video:
    # Create output directory
    Path(output_video_directory).mkdir(parents=True, exist_ok=True)

    convert_frames_to_mp4(frames, os.path.join(output_video_directory, f"{uuid_prefix}.mp4"), fps=fps)

  if not no_save_gif:
    # Create output directory
    Path(output_gif_directory).mkdir(parents=True, exist_ok=True)

    # Convert frames to GIF
    convert_frames_to_gif(frames, os.path.join(output_gif_directory, f"{uuid_prefix}.gif"), duration=1000 // fps)


if __name__ == "__main__":
  argsparser = argparse.ArgumentParser()
  argsparser.add_argument("--prompt", type=str, default="")
  argsparser.add_argument("--negative_prompt", type=str, default="")
  argsparser.add_argument("--frame_count", type=int, default=16)
  argsparser.add_argument("--num_inference_steps", type=int, default=20)
  argsparser.add_argument("--guidance_scale", type=float, default=7.5)
  argsparser.add_argument("--image_guidance_scale", type=float, default=1.5)
  argsparser.add_argument("--width", type=int, default=512)
  argsparser.add_argument("--height", type=int, default=512)
  argsparser.add_argument("--dtype", type=str, default="float16")
  argsparser.add_argument("--output_frames_directory", type=str, default="output_frames")
  argsparser.add_argument("--output_video_directory", type=str, default="output_videos")
  argsparser.add_argument("--output_gif_directory", type=str, default="output_gifs")
  argsparser.add_argument("--init_image", type=str, default=None)
  argsparser.add_argument("--init_folder", type=str, default=None)
  argsparser.add_argument("--motion_module", type=str, default="checkpoints/viddle-pix2pix-animatediff-v1.ckpt")
  argsparser.add_argument("--t2i_model", type=str, default="timbrooks/instruct-pix2pix")
  argsparser.add_argument("--seed", type=int, default=42)
  argsparser.add_argument("--fps", type=int, default=15)
  argsparser.add_argument("--no_save_frames", action="store_true", default=False)
  argsparser.add_argument("--no_save_video", action="store_true", default=False)
  argsparser.add_argument("--no_save_gif", action="store_true", default=False)
  args = argsparser.parse_args()

  run(t2i_model=args.t2i_model,
      prompt=args.prompt, 
      negative_prompt=args.negative_prompt, 
      frame_count=args.frame_count,
      num_inference_steps=args.num_inference_steps,
      guidance_scale=args.guidance_scale,
      width=args.width,
      height=args.height,
      dtype=args.dtype,
      output_frames_directory=args.output_frames_directory,
      output_video_directory=args.output_video_directory,
      output_gif_directory=args.output_gif_directory,
      motion_module=args.motion_module,
      init_image=args.init_image,
      init_folder=args.init_folder,
      seed=args.seed,
      fps=args.fps,
      no_save_frames=args.no_save_frames,
      no_save_video=args.no_save_video,
      no_save_gif=args.no_save_gif,
      )

  


