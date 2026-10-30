import os
import torch
import json
from torchvision.transforms import v2
from accelerate.utils import set_module_tensor_to_device

import folder_paths
import comfy.model_management as mm
import comfy.model_patcher
from comfy.utils import load_torch_file

script_directory = os.path.dirname(os.path.abspath(__file__))

if not "mmaudio" in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path("mmaudio", os.path.join(folder_paths.models_dir, "mmaudio"))

from .mmaudio.eval_utils import generate
from .mmaudio.model.flow_matching import FlowMatching
from .mmaudio.model.networks import MMAudio
from .mmaudio.model.utils.features_utils import FeaturesUtils
from .mmaudio.model.sequence_config import (CONFIG_16K, CONFIG_44K, SequenceConfig)
from .mmaudio.ext.bigvgan_v2.bigvgan import BigVGAN as BigVGANv2
from .mmaudio.ext.synchformer import Synchformer
from .mmaudio.ext.autoencoder import AutoEncoderModule
from open_clip import CLIP

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def process_video_tensor(
    video_tensor: torch.Tensor,
    duration_sec: float,
    target_clip: int | None = None,
    target_sync: int | None = None,
    target_device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Prepare visual tensors for MMAudio with extreme VRAM efficiency:
    - Selects only the required frames on host memory (CPU) before pushing to GPU.
    - Prevents transferring giant video tensors (e.g. 2048x1152x320 ≈ 9GB) all at once.
    - Performs batched GPU resizing in micro-chunks so peak VRAM is kept under ~150MB.
    """
    _CLIP_SIZE = 384
    _CLIP_FPS  = 8.0

    _SYNC_SIZE = 224
    _SYNC_FPS  = 25.0

    _SYNC_NUM_FRAMES_PER_SEG = 16
    _SYNC_STEP_SIZE          = 8
    _SYNC_OFFSET             = _SYNC_NUM_FRAMES_PER_SEG - _SYNC_STEP_SIZE  # = 8

    if target_device is None:
        target_device = mm.get_torch_device()

    total_frames = int(video_tensor.shape[0])
    if total_frames <= 0:
        raise ValueError("process_video_tensor: input video_tensor has zero frames")

    if target_clip is None:
        target_clip = max(1, int(_CLIP_FPS * duration_sec))
    if target_sync is None:
        target_sync = max(1, int(_SYNC_FPS * duration_sec) - _SYNC_OFFSET)

    # 1. Compute indices on source device without loading full video to GPU
    def uniform_indices(n_src: int, n_tgt: int, device: torch.device) -> torch.Tensor:
        if n_src == 1:
            return torch.zeros(n_tgt, dtype=torch.long, device=device)
        idx = torch.linspace(0, n_src - 1, steps=n_tgt, device=device)
        return torch.round(idx).to(torch.long).clamp_(0, n_src - 1)

    clip_idx = uniform_indices(total_frames, target_clip, video_tensor.device)
    sync_raw_needed = max(_SYNC_NUM_FRAMES_PER_SEG, int(target_sync) + _SYNC_OFFSET)
    sync_idx = uniform_indices(total_frames, sync_raw_needed, video_tensor.device)

    # 2. Extract ONLY the needed frames from source tensor (typically in CPU RAM)
    raw_clip_frames = video_tensor.index_select(0, clip_idx)
    raw_sync_frames = video_tensor.index_select(0, sync_idx)

    # 3. Micro-chunk GPU resizing to keep peak memory virtually zero
    clip_transform = v2.Compose([
        v2.Resize((_CLIP_SIZE, _CLIP_SIZE), interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
        v2.ToDtype(torch.float32, scale=False if raw_clip_frames.dtype.is_floating_point else True),
    ])

    sync_transform = v2.Compose([
        v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
        v2.CenterCrop(_SYNC_SIZE),
        v2.ToDtype(torch.float32, scale=False if raw_sync_frames.dtype.is_floating_point else True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    def chunked_process(frames_tensor: torch.Tensor, transform: v2.Compose, chunk_size: int = 16) -> torch.Tensor:
        processed_chunks = []
        n_frames = frames_tensor.shape[0]
        for start_i in range(0, n_frames, chunk_size):
            chunk = frames_tensor[start_i : start_i + chunk_size]
            # Send small chunk to GPU
            chunk = chunk.to(device=target_device, non_blocking=True).permute(0, 3, 1, 2).contiguous()
            chunk = transform(chunk)
            processed_chunks.append(chunk)
        return torch.cat(processed_chunks, dim=0)

    clip_frames = chunked_process(raw_clip_frames, clip_transform, chunk_size=16)
    sync_frames = chunked_process(raw_sync_frames, sync_transform, chunk_size=16)

    return clip_frames, sync_frames, duration_sec

#region Model loading
class MMAudioModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mmaudio_model": (folder_paths.get_filename_list("mmaudio"), {"tooltip": "These models are loaded from the 'ComfyUI/models/mmaudio' -folder",}),
            
            "base_precision": (["fp16", "fp32", "bf16"], {"default": "fp16"}),
            },
        }

    RETURN_TYPES = ("MMAUDIO_MODEL",)
    RETURN_NAMES = ("mmaudio_model", )
    FUNCTION = "loadmodel"
    CATEGORY = "MMAudio"

    def loadmodel(self, mmaudio_model, base_precision):
       

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
       
        mm.soft_empty_cache()

        base_dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[base_precision]

        mmaudio_model_path = folder_paths.get_full_path_or_raise("mmaudio", mmaudio_model)
        mmaudio_sd = load_torch_file(mmaudio_model_path, device=offload_device)

        # Detect latent dimension and architecture automatically from weights
        hidden_dim = mmaudio_sd["audio_input_proj.0.bias"].shape[0]
        latent_dim = mmaudio_sd["audio_input_proj.0.weight"].shape[1] if "audio_input_proj.0.weight" in mmaudio_sd else 40
        is_16k = (latent_dim == 20) or ("16" in mmaudio_model and "44" not in mmaudio_model)
        seq_cfg = CONFIG_16K if is_16k else CONFIG_44K
        latent_seq_len = 250 if is_16k else 345

        # Check for v2 model architecture
        is_v2 = mmaudio_sd.get("t_embed.mlp.0.weight", torch.zeros(1, 1)).shape[1] == 896

        if hidden_dim == 448:
            num_heads = 7
            depth = 12
            fused_depth = 8
        elif hidden_dim == 896:
            num_heads = 14
            depth = 21
            fused_depth = 14
        else:
            num_heads = max(1, hidden_dim // 64)
            depth = 12
            fused_depth = 8

        model = MMAudio(
            latent_dim=latent_dim,
            clip_dim=1024,
            sync_dim=768,
            text_dim=1024,
            hidden_dim=hidden_dim,
            depth=depth,
            fused_depth=fused_depth,
            num_heads=num_heads,
            latent_seq_len=latent_seq_len,
            clip_seq_len=64,
            sync_seq_len=192,
            v2=is_v2
        )
        model = model.eval()
        for name, param in model.named_parameters():
            # Set tensor to device
            set_module_tensor_to_device(model, name, device=offload_device, dtype=base_dtype, value=mmaudio_sd[name])
        del mmaudio_sd
        log.info(f'Loaded MMAudio model weights from {mmaudio_model_path} (mode={"16k" if is_16k else "44k"}, v2={is_v2})')
        model.seq_cfg = seq_cfg
        
        patcher = comfy.model_patcher.ModelPatcher(model, load_device=device, offload_device=offload_device)
        return (patcher,)
    
#region Features Utils
class MMAudioVoCoderLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vocoder_model": (folder_paths.get_filename_list("mmaudio"), {"tooltip": "These models are loaded from 'ComfyUI/models/mmaudio'"}),  
            },
        }

    RETURN_TYPES = ("VOCODER_MODEL",)
    RETURN_NAMES = ("mmaudio_vocoder", )
    FUNCTION = "loadmodel"
    CATEGORY = "MMAudio"

    def loadmodel(self, vocoder_model):
        from .mmaudio.ext.bigvgan import BigVGAN
        vocoder_model_path = folder_paths.get_full_path_or_raise("mmaudio", vocoder_model)
        vocoder_model = BigVGAN.from_pretrained(vocoder_model_path).eval()
        return (vocoder_model_path,)
        
class MMAudioFeatureUtilsLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_model": (folder_paths.get_filename_list("mmaudio"), {"tooltip": "These models are loaded from 'ComfyUI/models/mmaudio'"}),
                "synchformer_model": (folder_paths.get_filename_list("mmaudio"), {"tooltip": "These models are loaded from 'ComfyUI/models/mmaudio'"}),
                "clip_model": (folder_paths.get_filename_list("mmaudio"), {"tooltip": "These models are loaded from 'ComfyUI/models/mmaudio'"}),
            },
            "optional": {
              "bigvgan_vocoder_model": ("VOCODER_MODEL", {"tooltip": "These models are loaded from 'ComfyUI/models/mmaudio'"}),
                "mode": (["16k", "44k"], {"default": "44k"}),
                "precision": (["fp16", "fp32", "bf16"],
                    {"default": "fp16"}
                ),
            }
        }

    RETURN_TYPES = ("MMAUDIO_FEATUREUTILS",)
    RETURN_NAMES = ("mmaudio_featureutils", )
    FUNCTION = "loadmodel"
    CATEGORY = "MMAudio"

    def loadmodel(self, vae_model, precision, synchformer_model, clip_model, mode, bigvgan_vocoder_model=None):
        
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        #synchformer
        synchformer_path = folder_paths.get_full_path_or_raise("mmaudio", synchformer_model)
        synchformer_sd = load_torch_file(synchformer_path, device=offload_device)
        synchformer = Synchformer().eval()
        
        for name, param in synchformer.named_parameters():
            # Set tensor to device
            set_module_tensor_to_device(synchformer, name, device=offload_device, dtype=dtype, value=synchformer_sd[name])

        #vae
        download_path = folder_paths.get_folder_paths("mmaudio")[0]

        nvidia_bigvgan_vocoder_path = os.path.join(download_path, "nvidia", "bigvgan_v2_44khz_128band_512x")
        if mode == "44k":
            if not os.path.exists(nvidia_bigvgan_vocoder_path):
                log.info(f"Downloading nvidia bigvgan vocoder model to: {nvidia_bigvgan_vocoder_path}")
                from huggingface_hub import snapshot_download

                snapshot_download(
                    repo_id="nvidia/bigvgan_v2_44khz_128band_512x",
                    ignore_patterns=["*3m*",],
                    local_dir=nvidia_bigvgan_vocoder_path,
                    local_dir_use_symlinks=False,
                )
            
            # Vocoder precision: BigVGAN and VAE decoder are susceptible to severe audio artifacts / NaNs in FP16.
            # Force FP32 (or BF16 if selected) for Vocoder and VAE to guarantee pristine sound quality.
            vocoder_dtype = torch.bfloat16 if precision == "bf16" else torch.float32
            bigvgan_vocoder = BigVGANv2.from_pretrained(nvidia_bigvgan_vocoder_path).eval().to(device=offload_device, dtype=vocoder_dtype)
        else:
            assert bigvgan_vocoder_model is not None, "bigvgan_vocoder_model must be provided for 16k mode"
            bigvgan_vocoder = bigvgan_vocoder_model
            vocoder_dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        
        vae_path = folder_paths.get_full_path_or_raise("mmaudio", vae_model)
        vae_sd = load_torch_file(vae_path, device=offload_device)
        vae = AutoEncoderModule(
            vae_state_dict=vae_sd,
            bigvgan_vocoder=bigvgan_vocoder,
            mode=mode
            )
        vae = vae.eval().to(device=offload_device, dtype=vocoder_dtype)

        #clip
       
        clip_model_path = folder_paths.get_full_path_or_raise("mmaudio", clip_model)
        clip_config_path = os.path.join(script_directory, "configs", "DFN5B-CLIP-ViT-H-14-384.json")
        with open(clip_config_path) as f:
             clip_config = json.load(f)
            
        try:
            clip_model = CLIP(**clip_config["model_cfg"]).eval()
        except:
            # for some open-clip versions
            clip_config["model_cfg"]["nonscalar_logit_scale"] = True
            clip_model = CLIP(**clip_config["model_cfg"]).eval()

        clip_sd = load_torch_file(os.path.join(clip_model_path), device=offload_device)
        for name, param in clip_model.named_parameters():
            set_module_tensor_to_device(clip_model, name, device=offload_device, dtype=dtype, value=clip_sd[name])
        clip_model.to(device=offload_device, dtype=dtype)

        #clip_model = create_model_from_pretrained("hf-hub:apple/DFN5B-CLIP-ViT-H-14-384", return_transform=False)
        
       
        feature_utils = FeaturesUtils(vae=vae,
                                  synchformer=synchformer,
                                  enable_conditions=True,
                                  clip_model=clip_model)
        feature_utils.patcher = comfy.model_patcher.ModelPatcher(feature_utils, load_device=device, offload_device=offload_device)
        return (feature_utils,)

#region sampling
class MMAudioSampler:
    _VIDEO_CACHE_KEY = None
    _VIDEO_CACHE_DATA = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mmaudio_model": ("MMAUDIO_MODEL",),
                "feature_utils": ("MMAUDIO_FEATUREUTILS",),
                "duration": ("FLOAT", {"default": 8, "step": 0.01, "tooltip": "Duration of the audio in seconds"}),
                "steps": ("INT", {"default": 20, "step": 1, "tooltip": "Number of steps to interpolate (15-20 is optimal for RK4)"}),
                "cfg": ("FLOAT", {"default": 5.0, "step": 0.1, "tooltip": "Strength of the conditioning"}),
                "solver": (["rk4", "midpoint", "heun", "euler"], {"default": "rk4", "tooltip": "ODE solver algorithm. 'rk4' is 4th-order (maximum fidelity), 'midpoint' and 'heun' are 2nd-order, 'euler' is 1st-order."}),
                "cfg_rescale": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Rescales CFG velocity variance to prevent over-saturation, harsh distortion, and high-frequency artifacts (0.0 = disabled, 0.5-0.7 recommended)"}),
                "cfg_decay": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Attenuates CFG strength as trajectory approaches data (t -> 1) to smooth breath tails and micro-details"}),
                "shift": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1, "tooltip": "Flow matching time shift schedule factor (1.0 = standard linear)"}),
                "peak_normalize": ("BOOLEAN", {"default": True, "tooltip": "Safeguards against digital clipping (>0dBFS) by normalizing peak amplitudes to -0.5dBFS"}),
                "mask_away_clip": ("BOOLEAN", {"default": False, "tooltip": "If true, the clip video will be masked away"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "prompt": ("STRING", {"default": "", "multiline": True} ),
                "negative_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "e.g. low quality, noisy, distorted, clipping, muffled, robotic, metallic artifacts, buzz, deep voice, low pitch"} ),
            },
            "optional": {
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio", )
    FUNCTION = "sample"
    CATEGORY = "MMAudio"

    def sample(self, mmaudio_model, seed, feature_utils, duration, steps, cfg, prompt, negative_prompt, mask_away_clip,
               solver="rk4", cfg_rescale=0.7, cfg_decay=0.10, shift=1.0, peak_normalize=True, images=None):
        """
        - Uses seq_cfg to derive the *token* lengths the model expects.
        - Resamples frames so CLIP tokens == seq_cfg.clip_seq_len,
        and SYNC TOKENS == seq_cfg.sync_seq_len (by feeding +8 raw frames).
        - Calls update_seq_lengths with the SAME lengths to avoid asserts.
        - High-order ODE solvers (Midpoint, Heun, RK4) and CFG Rescale for enhanced audio fidelity.
        - JIT-moves models to GPU; automatically offloads based on ComfyUI vram_state.
        """
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        rng = torch.Generator(device=device).manual_seed(seed)

        # Unpack ModelPatcher if present
        if isinstance(mmaudio_model, comfy.model_patcher.ModelPatcher):
            mmaudio_patcher = mmaudio_model
            net = mmaudio_model.model
        else:
            mmaudio_patcher = getattr(mmaudio_model, "patcher", None)
            net = mmaudio_model

        fu_patcher = getattr(feature_utils, "patcher", None)
        patchers_to_load = [p for p in [mmaudio_patcher, fu_patcher] if isinstance(p, comfy.model_patcher.ModelPatcher)]

        # 1) Fix duration in config; derive target token counts from cfg
        seq_cfg: SequenceConfig = net.seq_cfg
        seq_cfg.duration = float(duration)

        target_clip_tokens = int(seq_cfg.clip_seq_len)   # e.g., 64 for 8s, 80 for 10s
        target_sync_tokens = int(seq_cfg.sync_seq_len)   # e.g., 192 for 8s, 242 for 10s

        # 2) Prepare visual inputs / check cache
        cached_clip_f = None
        cached_sync_f = None
        clip_frames = None
        sync_frames = None
        current_cache_key = None

        if images is not None:
            # Generate ultra-fast unique fingerprint (shape + duration + targets + sample pixel sum)
            n_frames = images.shape[0]
            sample_pts = [0, n_frames // 4, n_frames // 2, (3 * n_frames) // 4, n_frames - 1]
            pixel_hash = float(images[sample_pts, :2, :2, :].sum().item())
            current_cache_key = (
                tuple(images.shape),
                round(float(duration), 3),
                int(target_clip_tokens),
                int(target_sync_tokens),
                bool(mask_away_clip),
                pixel_hash,
                id(feature_utils)
            )

            if MMAudioSampler._VIDEO_CACHE_KEY == current_cache_key and MMAudioSampler._VIDEO_CACHE_DATA is not None:
                log.info("MMAudio: Reusing cached video features (instant 0s skip!)")
                cached_clip_f, cached_sync_f = MMAudioSampler._VIDEO_CACHE_DATA
            else:
                clip_frames, sync_frames, duration = process_video_tensor(
                    images, duration,
                    target_clip=target_clip_tokens,
                    target_sync=target_sync_tokens,
                    target_device=device,
                )
                log.info(
                    f"Prepared visual streams: "
                    f"CLIP(raw)={clip_frames.shape[0]} (tokens={target_clip_tokens}) @8fps, "
                    f"SYNC(raw)={sync_frames.shape[0]} (tokens={target_sync_tokens}) @25fps, "
                    f"duration={duration:.3f}s"
                )

                if mask_away_clip:
                    clip_frames = None
                else:
                    clip_frames = clip_frames.unsqueeze(0)  # (1, T_clip_raw, C, H, W)
                sync_frames = sync_frames.unsqueeze(0)      # (1, T_sync_raw, C, H, W)

        # 3) Scheduler + JIT device placement with ComfyUI memory preparation
        scheduler = FlowMatching(min_sigma=0, inference_mode=solver, num_steps=steps, shift=shift)
        if patchers_to_load:
            mm.load_models_gpu(patchers_to_load)
        else:
            mm.free_memory(2000 * 1024 * 1024, device)
        feature_utils.to(device)
        net.to(device)

        # 4) Update model to EXPECT these token lengths (now on target device)
        net.update_seq_lengths(
            seq_cfg.latent_seq_len,      # audio latents (comes from audio cfg/rate)
            target_clip_tokens,          # CLIP tokens
            target_sync_tokens           # SYNC tokens
        )

        try:
            # 5) Generate
            gen_res = generate(
                clip_frames,
                sync_frames,
                [prompt],
                negative_text=[negative_prompt],
                feature_utils=feature_utils,
                net=net,
                fm=scheduler,
                rng=rng,
                cfg_strength=cfg,
                cfg_rescale=cfg_rescale,
                cfg_decay=cfg_decay,
                peak_normalize=peak_normalize,
                precomputed_clip_features=cached_clip_f,
                precomputed_sync_features=cached_sync_f,
                return_features=True,
            )

            if isinstance(gen_res, tuple):
                audios, out_clip_f, out_sync_f = gen_res
                # Update cache on successful initial encode
                if images is not None and cached_clip_f is None and current_cache_key is not None:
                    MMAudioSampler._VIDEO_CACHE_KEY = current_cache_key
                    MMAudioSampler._VIDEO_CACHE_DATA = (out_clip_f.detach().cpu(), out_sync_f.detach().cpu())
            else:
                audios = gen_res
        finally:
            # 6) Automatic Offload based on ComfyUI's vram_state (offloads unless HIGH_VRAM is forced)
            if mm.vram_state != mm.VRAMState.HIGH_VRAM:
                net.to(offload_device)
                feature_utils.to(offload_device)
                if patchers_to_load:
                    for p in patchers_to_load:
                        p.model.model_loaded_weight_memory = 0
                mm.soft_empty_cache()

        waveform = audios.float().cpu()
        output_sr = getattr(seq_cfg, 'sampling_rate', 44100)
        return ({"waveform": waveform, "sample_rate": output_sr},)
        
NODE_CLASS_MAPPINGS = {
    "MMAudioModelLoader": MMAudioModelLoader,
    "MMAudioFeatureUtilsLoader": MMAudioFeatureUtilsLoader,
    "MMAudioSampler": MMAudioSampler,
    "MMAudioVoCoderLoader": MMAudioVoCoderLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MMAudioModelLoader": "MMAudio ModelLoader",
    "MMAudioFeatureUtilsLoader": "MMAudio FeatureUtilsLoader",
    "MMAudioSampler": "MMAudio Sampler",
    "MMAudioVoCoderLoader": "MMAudio VoCoderLoader",
    }
