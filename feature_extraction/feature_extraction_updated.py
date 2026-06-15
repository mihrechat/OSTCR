import torch
import torch.nn.functional as F
from timm.models import create_model
import numpy as np
import os
from PIL import Image, ImageSequence
import json
import shutil
from transformers import TimesformerModel
from matplotlib import pyplot as plt
import os

os.environ["HF_HUB_OFFLINE"] = "1"



# please download swin, resnet and timesformer pretrained models
swin_path = '/root/autodl-tmp/swin/swin_base_patch4_window7_224.pth'
time_path = '/root/autodl-tmp/model/models--facebook--timesformer-base-finetuned-k400/snapshots/8aaf40ea7d3d282dcb0a5dea01a198320d15d6c0'


def visualize_object_mask(frame, mask, activation, obj_id=None, alpha=0.4):
    frame_np = np.array(frame)
    mask = mask.astype(bool)
    overlay = frame_np.copy()
    
    # Red overlay for mask
    overlay[mask] = (0.7 * overlay[mask] + np.array([255, 0, 0]) * alpha).astype(np.uint8)
    
    # Save mask visualization
    plt.figure(figsize=(5, 5))
    plt.imshow(overlay)
    plt.axis("off")
    plt.savefig(f"object_{obj_id}.png", bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Save activation map
    plt.figure(figsize=(5, 5))
    plt.imshow(activation, alpha=0.5, cmap="jet")
    plt.axis("off")
    plt.savefig(f"activation_{obj_id}.png", bbox_inches='tight', pad_inches=0)
    plt.close()
    
    
def feature_activation_map(feature_map):

    # feature_map: (H, W, C)
    activation = feature_map.mean(dim=-1)

    activation = activation.cpu().numpy()

    activation = (activation - activation.min()) / (activation.max() - activation.min() + 1e-6)

    return activation
    
def load_gif_frames(path):
    img = Image.open(path)
    frames = [
        np.array(f.convert("RGB"))
        for f in ImageSequence.Iterator(img)
    ]
    return frames

def load_gif_frames_pil(path):
    img = Image.open(path)
    frames = [
        frame.convert("RGB")   # keep as PIL Image
        for frame in ImageSequence.Iterator(img)
    ]
    return frames

def get_frames(video_path):
    if video_path.endswith(".gif"):
        return load_gif_frames_pil(video_path)
    else:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        return [Image.fromarray(vr[i].asnumpy()) for i in range(len(vr))]


import os
import json
import gc
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
from PIL import Image
from timm.models import create_model
from torchvision import transforms
import pycocotools.mask as mask_utils

import torch
import torch.nn.functional as F
import timm
import numpy as np

from PIL import Image
from typing import List, Dict
from torchvision import transforms
from transformers import TimesformerModel, TimesformerConfig


class HybridVideoFeatureExtractor:

    def __init__(
        self,
        swin_model_name: str = swin_path,
        timesformer_path: str = time_path,
        input_resolution: int = 224,
        num_clips: int = 8,
        frames_per_clip: int = 12,
    ):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        self.return_swin_layers = True
        self.patch_size = 4
        self.num_clips = num_clips
        self.frames_per_clip = frames_per_clip

        self.swin_features = {}
        
        print("\n🔧 Loading Swin visual backbone...")

        self.swin = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=False,
            num_classes=0,
        ).to(self.device)
        
        checkpoint = torch.load(swin_model_name)
        
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]

        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        
        checkpoint = {
            k.replace("module.", ""): v
            for k, v in checkpoint.items()
        }

        missing, unexpected = self.swin.load_state_dict(
            checkpoint,
            strict=False
        )

        print("Missing keys:", len(missing))
        print("Unexpected keys:", len(unexpected))
        self.swin.eval()

        for p in self.swin.parameters():
            p.requires_grad = False
            

        def hook_fn(name):

            def fn(module, input, output):

                feat = output.detach()

                # ------------------------------------------------------
                # Convert token format -> spatial format
                # ------------------------------------------------------

                if feat.ndim == 3:

                    B, L, C = feat.shape

                    H = W = int(L ** 0.5)

                    feat = feat.view(B, H, W, C)

                self.swin_features[name] = feat

            return fn
        
        self.swin.layers[0].register_forward_hook(
            hook_fn("layer1")
        )

        self.swin.layers[1].register_forward_hook(
            hook_fn("layer2")
        )

        self.swin.layers[2].register_forward_hook(
            hook_fn("layer3")
        )

        self.swin.layers[3].register_forward_hook(
            hook_fn("layer4")
        )

        # ==========================================================
        # SWIN VISUAL BACKBONE
        # ==========================================================

        self.visual_dim = self.swin.num_features
        

        # ==========================================================
        # TIMESFORMER MOTION BACKBONE
        # ==========================================================

        print(" Loading TimeSformer motion backbone...")

        # config = TimesformerConfig.from_pretrained(
        #     timesformer_path
        # )
        

        self.timesformer = TimesformerModel.from_pretrained(
            timesformer_path,
            local_files_only=True,
        ).to(self.device)

        self.timesformer.eval()

        for p in self.timesformer.parameters():
            p.requires_grad = False
            
        self.motion_dim = self.timesformer.config.hidden_size

        # ==========================================================
        # TRANSFORMS
        # ==========================================================

        self.visual_transform = transforms.Compose([
            transforms.Resize(
                (input_resolution, input_resolution),
                Image.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.motion_transform = transforms.Compose([
            transforms.Resize(
                (input_resolution, input_resolution),
                Image.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.45, 0.45, 0.45],
                std=[0.225, 0.225, 0.225]
            )
        ])

        print(f"\n Hybrid extractor ready!")
        print(f"   Visual dim : {self.visual_dim}")
        print(f"   Motion dim : {self.motion_dim}")
        print(f"   Num clips  : {num_clips}")
        print(f"   Device     : {self.device}\n")

    # ==============================================================
    # MAIN EXTRACTION
    # ==============================================================
    
    def extract_features(
        self,
        pil_frames: List[Image.Image],
    ) -> Dict[str, np.ndarray]:

        with torch.no_grad():

            # ------------------------------------------------------
            # VISUAL FEATURES (FRAME-LEVEL)
            # ------------------------------------------------------
            visual_outputs = self.extract_visual_features(pil_frames)

            # ------------------------------------------------------
            # MOTION FEATURES (CLIP-LEVEL)
            # ------------------------------------------------------

            motion_features = self.extract_motion_features(pil_frames)
                
        return {

                # Main visual feature

                "layer1":
                    visual_outputs["layer1"].cpu().numpy().astype(np.float16)
                    if "layer1" in visual_outputs else None,

                "layer2":
                    visual_outputs["layer2"].cpu().numpy().astype(np.float16)
                    if "layer2" in visual_outputs else None,

                "layer3":
                    visual_outputs["layer3"].cpu().numpy().astype(np.float16)
                    if "layer3" in visual_outputs else None,

                "layer4":
                    visual_outputs["layer4"].cpu().numpy().astype(np.float16)
                    if "layer4" in visual_outputs else None,

                # Motion branch
                "motion_features":
                    motion_features.cpu().numpy().astype(np.float16),

            }

   

    # ==============================================================
    # VISUAL FEATURES (SWIN 2D)
    # ==============================================================

    # def extract_visual_features(
    #     self,
    #     pil_frames, 
    # ):

    #     frames_tensor = torch.stack([
    #         self.visual_transform(f)
    #         for f in pil_frames
    #     ], dim=0).to(self.device)

    #     self.swin_features.clear()

    #     _ = self.swin.forward_features(frames_tensor)

    #     outputs = {}

    #     for name, feat in self.swin_features.items():

    #         # timm swin:
    #         # [B, H, W, C]

    #         outputs[name] = feat.detach()

    #         # print(name, feat.shape)

    #     return outputs
    
    
    def extract_visual_features(
        self,
        pil_frames, 
    ):

        # transform all frames first (on CPU), then batch to GPU
        frames_tensor = torch.stack([
            self.visual_transform(f)
            for f in pil_frames
        ], dim=0)

        self.swin_features.clear()

        max_batch = 300
        num_frames = frames_tensor.shape[0]

        outputs = {}

        for start in range(0, num_frames, max_batch):
            end = min(start + max_batch, num_frames)
            batch = frames_tensor[start:end].to(self.device)

            self.swin_features.clear()
            _ = self.swin.forward_features(batch)

            for name, feat in self.swin_features.items():
                if name not in outputs:
                    outputs[name] = []
                outputs[name].append(feat.detach().cpu())

        # concatenate along batch dimension and move back to device if needed
        for name in outputs:
            outputs[name] = torch.cat(outputs[name], dim=0).to(self.device)

        return outputs

    # ==============================================================
    # MOTION FEATURES (TIMESFORMER)
    # ==============================================================

    def extract_motion_features(
        self,
        pil_frames
        ):

        total_frames = len(pil_frames)

        clip_features = []

        boundaries = np.linspace(
            0,
            total_frames,
            self.num_clips + 1,
            dtype=int
        )

        for i in range(self.num_clips):

            start = boundaries[i]
            end = boundaries[i + 1]

            clip = pil_frames[start:end]

            if len(clip) == 0:

                clip_features.append(
                    torch.zeros(self.motion_dim)
                )

                continue

            ids = np.linspace(
                0,
                len(clip) - 1,
                self.frames_per_clip,
                dtype=int
            )

            sampled = [clip[j] for j in ids]

            pixel_values = torch.stack([
                self.motion_transform(f)
                for f in sampled
            ], dim=0)

            # [1, T, 3, H, W]
            pixel_values = pixel_values.unsqueeze(0).to(
                self.device
            )

            outputs = self.timesformer(
                pixel_values=pixel_values
            )

            # CLS token
            
            tokens = outputs.last_hidden_state

            # Remove CLS token
            patch_tokens = tokens[:, 1:]

            # Mean and std over all spatial-temporal patches
            feat = patch_tokens.mean(dim=1)
            # std_feat = patch_tokens.std(dim=1)

            # feat = torch.cat(
            #     [mean_feat, std_feat],
            #     dim=-1
            # )
            

            clip_features.append(
                feat.squeeze(0).cpu()
            )

        motion_features = torch.stack(
            clip_features,
            dim=0
        )

        return motion_features

    
    def _preprocess_frames(self, pil_frames: List[Image.Image]) -> torch.Tensor:
        """
        Preprocess PIL frames for Swin3D.
        
        Returns:
            tensor: (1, 3, T, H, W) in range [0, 1]
        """
        H, W = self.input_resolution, self.input_resolution
        
        transform = transforms.Compose([
            transforms.Resize((H, W), Image.BILINEAR),
            transforms.ToTensor(),  # [0, 1]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        frames_list = [transform(frame) for frame in pil_frames]
        frames_tensor = torch.stack(frames_list, dim=1)  # (3, T, H, W)
        frames_tensor = frames_tensor.unsqueeze(0)  # (1, 3, T, H, W)
        
        return frames_tensor.to(self.device)


# ==================================================================
# UTILITIES: MASK HANDLING
# ==================================================================

def decode_rle_mask(rle) -> Optional[np.ndarray]:
    """Decode RLE mask to binary numpy array"""
    try:
        mask = mask_utils.decode(rle)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return mask.astype(np.uint8)
    except:
        return None



def _masked_pool(
    feature_map: torch.Tensor,   # [H, W, C]
    masks: np.ndarray,           # [N, H_orig, W_orig]
    eps: float = 1e-6,
    ):
    """
    Exact object pooling aligned with Swin feature resolution.

    Automatically resizes original masks to the exact
    feature map resolution before pooling.

    Returns:
        pooled: [N, C]
    """

    device = feature_map.device

    H_feat, W_feat, C = feature_map.shape
    N = masks.shape[0]

    # ----------------------------------------------------------
    # Resize ORIGINAL masks to exact feature resolution
    # ----------------------------------------------------------

    masks_t = torch.from_numpy(masks).float().to(device)

    # [N, 1, H_orig, W_orig]
    masks_t = masks_t.unsqueeze(1)

    # nearest preserves object boundaries
    masks_resized = F.interpolate(
        masks_t,
        size=(H_feat, W_feat),
        mode="nearest"
    )

    # [N, H_feat, W_feat]
    masks_resized = masks_resized.squeeze(1)

    # ----------------------------------------------------------
    # Flatten spatial dimensions
    # ----------------------------------------------------------

    feature_flat = feature_map.view(-1, C)

    # [N, H_feat*W_feat]
    masks_flat = masks_resized.view(N, -1)

    # ----------------------------------------------------------
    # Normalize masks
    # ----------------------------------------------------------

    masks_flat = masks_flat / (
        masks_flat.sum(dim=1, keepdim=True) + eps
    )

    # ----------------------------------------------------------
    # Weighted masked pooling
    # ----------------------------------------------------------

    pooled = masks_flat @ feature_flat

    return pooled.float()

def _extract_object_data(objects: List[Dict], pil_frames, frame_idx, activation) -> Tuple:
    """Extract object masks and metadata"""
    mask_list, obj_ids, bboxes, confs, classes = [], [], [], [], []
    
    activation = feature_activation_map(activation)
    
    for obj in objects:
        if "mask" not in obj:
            continue
        
        mask = decode_rle_mask(obj["mask"])
        if mask is None or mask.sum() == 0:
            continue
        
        H, W = mask.shape
        mask_list.append((mask > 0).astype(np.float32))
        obj_ids.append(obj["id"])
        bbox = np.concatenate([
            np.array(obj.get("bbox", [0, 0, 0, 0])),
            [H, W]
        ])
        bboxes.append(bbox)
        confs.append(float(obj.get("confidence", 1.0)))
        classes.append(obj.get("class_name", "unknown"))
        
    # for i in range(min(5, len(mask_list))):  # visualize first few objects

    #     visualize_object_mask(
    #         pil_frames[frame_idx],  # original frame
    #         mask_list[i],
    #         activation,
    #         obj_id=obj_ids[i]
    #     )
    
    return mask_list, obj_ids, bboxes, confs, classes


def to_python(obj):
    """Recursively convert numpy types to native Python for JSON"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python(v) for v in obj]
    return obj


# ==================================================================
# VIDEO LOADING (You need this utility)
# ==================================================================

def get_frames(video_path: str) -> List[Image.Image]:
    """
    Load all frames from video file.
    Supports: MP4, AVI, MOV, etc.
    
    You can use either:
    1. opencv-python (cv2)
    2. av (PyAV)
    3. decord
    
    Example with opencv:
    """
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frames.append(frame_pil)
        
        cap.release()
        return frames
    
    except ImportError:
        print("❌ OpenCV not installed. Install with: pip install opencv-python")
        raise


# ==================================================================
# MAIN FEATURE EXTRACTION: ALL FRAMES, NO SKIPPING
# ==================================================================

def process_video_all_frames(
    video_path: str,
    json_dir: str,
    save_root: str,
    video_name: str,
    feature_extractor: HybridVideoFeatureExtractor,
    k_objects: Optional[int] = None,
) -> None:
    
    
    os.makedirs(save_root, exist_ok=True)
    video_id = os.path.splitext(video_name)[0]
    save_dir = os.path.join(save_root, video_id)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"🎬 Processing: {video_id}")
    print(f"{'='*70}")

    # ── Save paths ────────────────────────────────────────────────
    node_layer1_path = os.path.join(save_dir, "node_layer1.npy")
    node_layer2_path = os.path.join(save_dir, "node_layer2.npy")
    node_layer3_path = os.path.join(save_dir, "node_layer3.npy")
    node_layer4_path = os.path.join(save_dir, "node_layer4.npy")
    motion_path = os.path.join(save_dir, "motion_features.npy")
    meta_path = os.path.join(save_dir, "node_metadata.json")

    # Skip if all exist
    all_exist = all([
        os.path.exists(node_layer1_path),
        os.path.exists(node_layer2_path),
        os.path.exists(node_layer3_path),
        os.path.exists(node_layer4_path),
        os.path.exists(motion_path),
        os.path.exists(meta_path),
    ])
    
    if all_exist:
        print(f" All features already exist. Skipping.")
        return

    # # ── Load video and metadata ────────────────────────────────────
    print(f"Loading video and metadata...")
    json_path = os.path.join(json_dir, video_id + ".json")

    # If JSON doesn't exist, remove video and return
    if not os.path.exists(json_path):
        print(f" JSON annotation not found for {video_id}. Skipping.")
        video_file_path = os.path.join(video_path, video_name)
        if os.path.exists(video_file_path):
            #use shutil.rmtree() to ensure deletion of video file
            shutil.rmtree(video_file_path, ignore_errors=True)
            if os.path.exists(video_file_path):
                os.remove(video_file_path)
            print(f"   Removed video file: {video_file_path}")
        return

    with open(json_path, "r") as f:
        graph = json.load(f)

    frame_indices = [int(idx) for idx in graph["frame_indices"]]

    # If frame indices are less than 1, remove both files
    if len(frame_indices) < 1:
        print(f" No valid frames in JSON for {video_id}. Skipping.")
        # Remove JSON
        if os.path.exists(json_path):
            shutil.rmtree(json_path, ignore_errors=True)
            os.remove(json_path)
            print(f"   Removed JSON file: {json_path}")
        # Remove video
        video_file_path = os.path.join(video_path, video_name)
        if os.path.exists(video_file_path):
            shutil.rmtree(video_file_path, ignore_errors=True)
            os.remove(video_file_path)
            print(f"   Removed video file: {video_file_path}")
        return
    # print(f"Loading video and metadata...")
    # json_path = os.path.join(json_dir, video_id + ".json")
    # #if json doesn't exists remove the video and return
    # if not os.path.exists(json_path):
    #     print(f"  JSON annotation not found for {video_id}. Skipping.")
    #     #remove the video file as well since it's not useful without annotation
    #     video_file_path = os.path.join(video_path, video_name)
    #     if os.path.exists(video_file_path):
    #         os.remove(video_file_path)
    #         # use rm in addition to os.remove to ensure deletion
    #         if os.path.exists(video_file_path):
    #             os.system(f"rm {video_file_path}")
    #         print(f"   Removed video file: {video_file_path}")
    #     return
    # with open(json_path, "r") as f:
    #     graph = json.load(f)
    
    

    # frame_indices = [int(idx) for idx in graph["frame_indices"]]
    
    # # if frame indices are less than 1, just remove the vide and json
    # if len(frame_indices) < 1:
    #     print(f"  No valid frames in JSON for {video_id}. Skipping.")
    #     # remove json
    #     if os.path.exists(json_path):
    #         os.remove(json_path)
    #         print(f"   Removed JSON file: {json_path}")
    #     # remove video
    #     video_file_path = os.path.join(video_path, video_name)
    #     if os.path.exists(video_file_path):
    #         os.remove(video_file_path)
    #         # use rm in addition to os.remove to ensure deletion
    #         if os.path.exists(video_file_path):
    #             os.system(f"rm {video_file_path}")
    #         print(f"   Removed video file: {video_file_path}")
    #     return
    frame_lookup = {int(fr["frame_idx"]): fr for fr in graph["frames"]}
    
    print(f"   Total frames in video from JSON: {len(frame_indices)}, {frame_indices[:5]}...")
    
    # Load all frames from video file
    print(f" Loading video frames from disk...")
    video_file_path = os.path.join(video_path, video_name)
    all_pil_frames = get_frames(video_file_path)
    print(f"   Loaded {len(all_pil_frames)} frames")

    # ── Phase 1: Extract multi-scale features for ALL frames ───────
    print(f"\n🔍 Extracting multi-scale features from ALL frames...")
    features_dict = feature_extractor.extract_features(all_pil_frames)
    
    # features_dict keys:
    # - 'layer1': (T, H1, W1, 128)
    # - 'layer2': (T, H2, W2, 256)
    # - 'layer3': (T, H3, W3, 512)
    # - 'layer4': (T, H4, W4, 1024)
    # - 'temporal_features': (T, 1024)

    # Save motion features
    print(f"\n Saving motion features...")
    motion_features = features_dict['motion_features']
    if not os.path.exists(motion_path):
        np.save(motion_path, motion_features)
        print(f"    motion_features.npy: shape {motion_features.shape}")

    # ── Phase 2: Per-frame, per-object feature extraction ──────────
    print(f"\n  Extracting per-object features...")
    
    node_layer1_list = []
    node_layer2_list = []
    node_layer3_list = []
    node_layer4_list = []
    
    node_frame_index, node_frame_id_map = [], []
    node_obj_ids_flat, node_bbox_flat, node_conf_flat, node_class_flat = [], [], [], []
    frame_node_counts = []

    # Convert features to torch tensors on GPU
    layer1_tensor = torch.from_numpy(features_dict['layer1']).float().to(feature_extractor.device)
    layer2_tensor = torch.from_numpy(features_dict['layer2']).float().to(feature_extractor.device)
    layer3_tensor = torch.from_numpy(features_dict['layer3']).float().to(feature_extractor.device)
    layer4_tensor = torch.from_numpy(features_dict['layer4']).float().to(feature_extractor.device)

    total_objects = 0

    # Process each frame
    for frame_idx, frame_id in enumerate(frame_indices):
        frame_id = int(frame_id)
        fr = frame_lookup.get(frame_id)
        
        if fr is None:
            frame_node_counts.append(0)
            continue

        curr_objects = fr["objects"]
        if k_objects is not None:
            curr_objects = curr_objects[:k_objects]

        # Extract object masks and metadata
        mask_list, obj_ids, obj_bboxes, obj_confs, obj_classes = _extract_object_data(curr_objects,all_pil_frames,frame_idx, layer1_tensor[frame_idx])

        if not mask_list:
            frame_node_counts.append(0)
            continue

        masks = np.stack(mask_list)  # (N, H, W)
        n_valid = len(obj_ids)

        # Get features for this frame (at all scales)
        cur_layer1 = layer1_tensor[frame_idx]  # (H1, W1, 128)
        cur_layer2 = layer2_tensor[frame_idx]  # (H2, W2, 256)
        cur_layer3 = layer3_tensor[frame_idx]  # (H3, W3, 512)
        cur_layer4 = layer4_tensor[frame_idx]  # (H4, W4, 1024)

        
        obj_feat_layer1 = _masked_pool(
            cur_layer1,
            masks
        )

        obj_feat_layer2 = _masked_pool(
            cur_layer2,
            masks
        )

        obj_feat_layer3 = _masked_pool(
            cur_layer3,
            masks
        )

        obj_feat_layer4 = _masked_pool(
            cur_layer4,
            masks
        )

        frame_node_counts.append(n_valid)
        node_layer1_list.append(obj_feat_layer1.cpu().numpy().astype(np.float16))
        node_layer2_list.append(obj_feat_layer2.cpu().numpy().astype(np.float16))
        node_layer3_list.append(obj_feat_layer3.cpu().numpy().astype(np.float16))
        node_layer4_list.append(obj_feat_layer4.cpu().numpy().astype(np.float16))

        node_frame_index.extend([frame_id] * n_valid)
        node_frame_id_map.extend([frame_idx] * n_valid)  # Index in features tensor
        node_obj_ids_flat.extend(obj_ids)
        node_bbox_flat.extend(obj_bboxes)
        node_conf_flat.extend(obj_confs)
        node_class_flat.extend(obj_classes)
        
        total_objects += n_valid

    # ── Stack features ────────────────────────────────────────────
    print(f"\n Stacking features...")
    
    node_layer1 = np.concatenate(node_layer1_list, axis=0) if node_layer1_list else np.array([], dtype=np.float16)
    node_layer2 = np.concatenate(node_layer2_list, axis=0) if node_layer2_list else np.array([], dtype=np.float16)
    node_layer3 = np.concatenate(node_layer3_list, axis=0) if node_layer3_list else np.array([], dtype=np.float16)
    node_layer4 = np.concatenate(node_layer4_list, axis=0) if node_layer4_list else np.array([], dtype=np.float16)

    print(f"   node_layer1: {node_layer1.shape}")
    print(f"   node_layer2: {node_layer2.shape}")
    print(f"   node_layer3: {node_layer3.shape}")
    print(f"   node_layer4: {node_layer4.shape}")

    # ── Save features ──────────────────────────────────────────────
    print(f"\n Saving features to disk...")
    
    if not os.path.exists(node_layer1_path):
        np.save(node_layer1_path, node_layer1)
        print(f"    node_layer1.npy saved")
    
    if not os.path.exists(node_layer2_path):
        np.save(node_layer2_path, node_layer2)
        print(f"    node_layer2.npy saved")
    
    if not os.path.exists(node_layer3_path):
        np.save(node_layer3_path, node_layer3)
        print(f"    node_layer3.npy saved")
    
    if not os.path.exists(node_layer4_path):
        np.save(node_layer4_path, node_layer4)
        print(f"   node_layer4.npy saved")

    # ── Save metadata ──────────────────────────────────────────────
    print(f"\n📝 Saving metadata...")
    
    metadata = {
        "video_id": video_id,
        "frame_node_counts": frame_node_counts,
        "node_frame_index": node_frame_index,
        "node_frame_id_map": node_frame_id_map,
        "node_obj_ids_flat": node_obj_ids_flat,
        "node_bbox": node_bbox_flat,
        "node_conf": node_conf_flat,
        "node_class": node_class_flat,
        "spatial_relations": graph.get("spatial_relations", {}),
        "temporal_relations": graph.get("temporal_relations", {}),
        "frame_indices": frame_indices,
        "total_objects": int(total_objects),
        "k_objects": k_objects,
    }

    with open(meta_path, "w") as f:
        json.dump(to_python(metadata), f, indent=4)
    
    print(f"    node_metadata.json saved")

    # ── Cleanup ────────────────────────────────────────────────────
    print(f"\n🧹 Cleaning up...")
    del features_dict, layer1_tensor, layer2_tensor, layer3_tensor, layer4_tensor
    del node_layer1_list, node_layer2_list, node_layer3_list, node_layer4_list
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f" COMPLETED: {video_id}")
    print(f"    Total objects: {total_objects}")
    print(f"    All {len(all_pil_frames)} frames processed (NO SKIPPING!)")
    print(f"    Full temporal resolution preserved!")
    print(f"{'='*70}\n")


# ==================================================================
# BATCH PROCESSING: PROCESS MULTIPLE VIDEOS
# ==================================================================

# def process_video_dataset(
#     video_path: str,
#     json_dir: str,
#     save_root: str,
#     feature_extractor: HybridVideoFeatureExtractor,
#     k_objects: Optional[int] = None,
#     max_videos: Optional[int] = None,
# ):
#     """
#     Process entire video dataset.
    
#     Args:
#         video_path: Directory with video files
#         json_dir: Directory with JSON annotations
#         save_root: Where to save features
#         feature_extractor: VideoSwinFeatureExtractor instance
#         k_objects: Limit objects per frame
#         max_videos: Stop after processing N videos (for testing)
#     """
    
#     video_files = sorted([
#         f for f in os.listdir(video_path)
#         if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))
#     ])
    
#     if max_videos:
#         video_files = video_files[:max_videos]
    
#     print(f"\n Starting batch processing...")
#     print(f"   Total videos to process: {len(video_files)}")
#     print(f"   Save directory: {save_root}\n")

#     for idx, video_name in enumerate(video_files, 1):
#         print(f"[{idx}/{len(video_files)}]", end=" ")
        
#         try:
#             process_video_all_frames(
#                 video_path=video_path,
#                 json_dir=json_dir,
#                 save_root=save_root,
#                 video_name=video_name,
#                 feature_extractor=feature_extractor,
#                 k_objects=k_objects,
#             )
#         except Exception as e:
#             print(f" ERROR: {str(e)}")
#             continue

#     print(f"\n{'='*70}")
#     print(f" BATCH PROCESSING COMPLETE!")
#     print(f"   Processed: {len(video_files)} videos")
#     print(f"   Features saved to: {save_root}")
#     print(f"{'='*70}\n")


def process_video_dataset(
    video_path: str,
    json_dir: str,
    save_root: str,
    feature_extractor: HybridVideoFeatureExtractor,
    k_objects=None,
    max_videos=None,
):
   
    # reference_check_dir_train = '/root/autodl-tmp/CausalSTGNet/datasets/open_ended_vqa/MSVD/train'
    # reference_check_dir_val = '/root/autodl-tmp/CausalSTGNet/datasets/open_ended_vqa/MSVD/val'
    # reference_check_dir_test = '/root/autodl-tmp/CausalSTGNet/datasets/open_ended_vqa/MSVD/test'
    
    video_files = sorted([
        f for f in os.listdir(video_path)
        if f.endswith(('.mp4', '.avi', '.mov', '.mkv', '.gif'))
    ])
    
    # allowed_names = {
    # d for d in os.listdir(reference_check_dir_test)
    # if os.path.isdir(os.path.join(reference_check_dir_test, d))
    # }
    # video_files_filtered = [
    # f for f in video_files
    # if os.path.splitext(f)[0] in allowed_names
    #     ]
    # print(f"Total videos found: {len(video_files)}")
    # print(f"Videos after filtering: {len(video_files_filtered)}")
    # print(f"Example video files: {video_files_filtered[:5]}")

    
    
    # if saved video folder name doesn't exist in json path remove the directory 
    
    # all_existing_saved_dirs = {
    #     d for d in os.listdir(save_root)        
    #     if os.path.isdir(os.path.join(save_root, d))
    # }
    # all_json_names = {
    #     os.path.splitext(f)[0] for f in os.listdir(json_dir)        
    #     if f.endswith('.json')
    # }
    
    # #verify the the first few names before remove:
    # print(f"Example existing saved dirs: length: {len(all_existing_saved_dirs)}")
    # print(f"Example json names: length: {len(all_json_names)}")
    
    # removed = 0
    # # if saved video folder name doesn't exist in json path remove the directory os.system(f'rm -rf {remove_dir_train}')
    # for saved_dir in all_existing_saved_dirs:
    #     if saved_dir not in all_json_names:
    #         remove_dir = os.path.join(save_root, saved_dir)
    #         os.system(f'rm -rf {remove_dir}') 
    #         removed += 1
    #         print(f"Removed saved directory since it doesn't exist in json path")
    # print(f"Total removed directories: {removed}")
    

    if max_videos:
        video_files = video_files[:max_videos]

    print(f"\n🎬 Starting batch processing...")
    print(f"   Total videos found: {len(video_files)}")
    print(f"   Save directory: {save_root}\n")

    processed = 0
    skipped = 0
    failed = 0

    for idx, video_name in enumerate(video_files, 1):

        video_id = os.path.splitext(video_name)[0]

        save_dir = os.path.join(save_root, video_id)

    #     # ------------------------------------------------------
    #     # REQUIRED OUTPUT FILES
    #     # ------------------------------------------------------

        required_files = [

            "node_layer1.npy",
            "node_layer2.npy",
            "node_layer3.npy",
            "node_layer4.npy",

            "motion_features.npy",

            "node_metadata.json",
        ]

        # ------------------------------------------------------
        # CHECK IF EVERYTHING EXISTS
        # ------------------------------------------------------

        already_done = True
        
        #Check if motoin_features.npy exists and the shape is correct with the expected shape (num_clips, motion_dim) = (24, 768*2)
      
        # if os.path.exists(os.path.join(save_dir, "motion_features.npy")):
            
        #     motion_feat_path = os.path.join(save_dir, "motion_features.npy")
        #     motion_feat = np.load(motion_feat_path)

        #     if motion_feat.shape != (num_clips, motion_dim ):
        #         print(f" Incorrect shape for motion_features.npy in {video_id}. Expected {(num_clips, motion_dim)}, got {motion_feat.shape}")
        #         #remove the folder that motion_features.npy is in since it is not correct
        #         os.remove(motion_feat_path)
        #         print(f'    Removed incorrect motion_features.npy for {video_id}')
        #         already_done = False
                
                
        if not os.path.exists(save_dir):
            already_done = False
        #if not in json path then skip it
        # if video_id not in all_json_names:
        #     #skip this video since we don't have the json file for it
        #     already_done = False
            

        else:

            for fname in required_files:

                full_path = os.path.join(
                    save_dir,
                    fname
                )

                if not os.path.exists(full_path):
                    already_done = False
                    break

    #     # ------------------------------------------------------
    #     # SKIP COMPLETED VIDEOS
    #     # ------------------------------------------------------

        if already_done:

            skipped += 1

            print(
                f"[{idx}/{len(video_files)}] "
                f" Skipping {video_id}"
            )

            continue

        # ------------------------------------------------------
        # PROCESS VIDEO
        # ------------------------------------------------------

        print(
            f"[{idx}/{len(video_files)}] "
            f" Processing {video_id}"
        )

        try:

            process_video_all_frames(
                video_path=video_path,
                json_dir=json_dir,
                save_root=save_root,
                video_name=video_name,
                feature_extractor=feature_extractor,
                k_objects=k_objects,
            )

            processed += 1

        except Exception as e:

            failed += 1

            print(
                f" ERROR in {video_id}: {str(e)}"
            )

            continue

    print(f"\n{'='*70}")
    print(" BATCH PROCESSING COMPLETE!")
    print(f"{'='*70}")

    print(f"Processed : {processed}")
    print(f"Skipped   : {skipped}")
    print(f"Failed    : {failed}")

    print(f"\n Features saved to:")
    print(save_root)

    print(f"{'='*70}\n")


if __name__ == "__main__":
    # ── CONFIGURATION ──────────────────────────────────────────────
    VIDEO_DIR_tgif = "/root/autodl-tmp/missing" 
    JSON_DIR_tgif = '/root/autodl-tmp/frame/train_masks' 
    SAVE_DIR_tgif = "/root/autodl-tmp/CausalSTGNet/datasets/multi-choice_vqa/TGIF/TGIF_frame/train"  
    VIDEO_DIR_msvd = "/root/autodl-tmp/YouTubeClips"
    JSON_DIR_msvd = "/root/autodl-tmp/CausalSTGNet/datasets/open_ended_vqa/MSVD/train2_masks"
    SAVE_DIR_msvd = "/root/autodl-tmp/CausalSTGNet/datasets/open_ended_vqa/MSVD_2/train"
    
    VIDEO_DIR = VIDEO_DIR_tgif
    JSON_DIR = JSON_DIR_tgif
    SAVE_DIR = SAVE_DIR_tgif
    # ── Initialize feature extractor ───────────────────────────────
    print("🚀 Initializing Video Swin Feature Extractor...\n")
    
    feature_extractor = HybridVideoFeatureExtractor(
                swin_model_name = swin_path,
                timesformer_path = time_path,
                input_resolution = 224,
                num_clips = 8,
                frames_per_clip = 12,
            )

    # ── Process videos ─────────────────────────────────────────────
    process_video_dataset(
        video_path=VIDEO_DIR,
        json_dir=JSON_DIR,
        save_root=SAVE_DIR,
        feature_extractor=feature_extractor,
        k_objects=None,  # Limit to top 20 objects per frame
        max_videos=None,  # None = all videos, or set to N for testing
    )

    print("\n Feature extraction pipeline complete!")
    
