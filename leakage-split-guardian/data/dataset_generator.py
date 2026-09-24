"""
dataset_generator.py
Synthetic Benchmark Dataset Generator for Leakage & Split Guardian (Topic C3).

Simulates a real-world multi-camera, video burst, route-based visual dataset:
- Multiple independent scenes/routes.
- 4 target classes: 'sedan', 'truck', 'motorcycle', 'emergency_vehicle'.
- 2 distinct forms of cross-split leakage:
    1. Temporal Burst Overlap: High-frequency frame burst from same camera & time window.
    2. Multi-Camera Spatial Overlap: Same target/event viewed simultaneously from multiple camera angles.
- True Independent Holdout Test Set (Unseen routes/scenes) to evaluate real-world generalization.
- Ground truth manifest & pairwise leak ground truth for precise F1/Recall/Precision evaluation.
"""

import os
import json
import random
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def create_synthetic_object_image(
    class_id: int,
    base_seed: int,
    variation_type: str = "base",
    cam_angle_offset: float = 0.0,
    time_offset: float = 0.0,
    img_size: tuple = (128, 128)
) -> Image.Image:
    """
    Generates a deterministic synthetic visual pattern representing a vehicle/object
    under camera perspective changes and temporal jitter.
    """
    rng = random.Random(base_seed)
    
    # Common vehicle paint finishes across all classes (silver, black, red, blue, white)
    paint_palettes = [
        (35, 35, 40),    # Obsidian Black
        (180, 185, 190), # Metallic Silver
        (190, 40, 35),   # Crimson
        (30, 80, 180),   # Deep Blue
        (230, 230, 235), # Polar White
    ]
    # Each burst event gets a realistic vehicle paint color
    paint_idx = (base_seed // 7) % len(paint_palettes)
    primary_col = paint_palettes[paint_idx]
    secondary_col = (max(0, primary_col[0] - 30), max(0, primary_col[1] - 30), max(0, primary_col[2] - 30))

    # Background lighting depends on scene and time
    scene_seed = (base_seed // 1000) * 1000
    bg_val = (scene_seed * 43) % 70 + 40
    bg_color = (bg_val, bg_val + 4, bg_val + 8)
    img = Image.new("RGB", img_size, color=bg_color)
    draw = ImageDraw.Draw(img)

    # Add scene context texture (road markings / pavement lines)
    road_y = int(img_size[1] * 0.65)
    draw.rectangle([0, road_y, img_size[0], img_size[1]], fill=(45, 45, 50))
    lane_x = ((scene_seed // 3) + int(time_offset * 12)) % img_size[0]
    draw.line([lane_x, road_y, lane_x, img_size[1]], fill=(210, 210, 170), width=3)

    # Class-specific geometry and structural features:
    # 0: Sedan (sleek, low height, wide)
    # 1: Truck (tall, boxy, heavy cargo bed)
    # 2: Motorcycle (very narrow, short, high clearance)
    # 3: Emergency Vehicle (box shape + dual flashing roof light beacons)
    cx = img_size[0] // 2 + int(time_offset * 6) + int(cam_angle_offset * 5)
    cy = int(img_size[1] * 0.58)

    if class_id == 0: # Sedan
        w, h = 64, 26
        cabin_w, cabin_h = 38, 14
    elif class_id == 1: # Truck
        w, h = 74, 46
        cabin_w, cabin_h = 30, 24
    elif class_id == 2: # Motorcycle
        w, h = 28, 36
        cabin_w, cabin_h = 16, 18
    else: # Emergency Vehicle
        w, h = 62, 34
        cabin_w, cabin_h = 42, 18

    # Draw object body
    bbox = [cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2]
    draw.rounded_rectangle(bbox, radius=4, fill=primary_col, outline=secondary_col, width=2)

    # Cabin / Top structure
    cabin_bbox = [cx - cabin_w // 2, cy - h // 2 - cabin_h + 2, cx + cabin_w // 2, cy - h // 2 + 2]
    cabin_color = (200, 220, 240) if class_id != 1 else (100, 100, 105)
    draw.rounded_rectangle(cabin_bbox, radius=3, fill=cabin_color, outline=secondary_col, width=1)

    # Emergency roof light beacon for class 3
    if class_id == 3:
        beacon_y = cy - h // 2 - cabin_h - 4
        draw.rectangle([cx - 10, beacon_y, cx - 2, beacon_y + 4], fill=(240, 20, 20))
        draw.rectangle([cx + 2, beacon_y, cx + 10, beacon_y + 4], fill=(20, 60, 240))
    
    # Wheels / Details
    wheel_r = 7
    wheel_y = cy + h // 2 - 2
    draw.ellipse([cx - w // 3 - wheel_r, wheel_y - wheel_r, cx - w // 3 + wheel_r, wheel_y + wheel_r], fill=(20, 20, 20))
    draw.ellipse([cx + w // 3 - wheel_r, wheel_y - wheel_r, cx + w // 3 + wheel_r, wheel_y + wheel_r], fill=(20, 20, 20))
    
    # Camera perspective noise or lighting shift
    if variation_type == "multi_camera":
        # Perspective shift simulation (tint & slight blur or sharpen)
        cam_tint = int(cam_angle_offset * 15)
        np_arr = np.array(img, dtype=np.int16)
        np_arr[:, :, 0] = np.clip(np_arr[:, :, 0] + cam_tint, 0, 255)
        np_arr[:, :, 2] = np.clip(np_arr[:, :, 2] - cam_tint, 0, 255)
        img = Image.fromarray(np_arr.astype(np.uint8))
        img = img.filter(ImageFilter.GaussianBlur(radius=0.4))
    elif variation_type == "temporal_burst":
        # Temporal burst: minor motion blur & pixel noise
        noise = np.random.RandomState(int(base_seed + time_offset * 100) % (2**31 - 1)).normal(0, 3, (img_size[1], img_size[0], 3))
        np_arr = np.clip(np.array(img, dtype=np.float32) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(np_arr)
        
    return img

def generate_benchmark_dataset(
    output_dir: str,
    num_trainval_scenes: int = 14,
    num_test_scenes: int = 4,
    burst_per_scene: int = 3,
    frames_per_burst: int = 4,
    num_cameras: int = 2,
    seed: int = 42
):
    """
    Creates complete benchmark dataset on disk:
    - Images organized by scene & camera
    - metadata.json with labels and cluster ground truth
    - ground_truth_leakage_pairs.json with all known true leakage pairs
    """
    random.seed(seed)
    np.random.seed(seed)
    
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    metadata = []
    ground_truth_clusters = {} # cluster_id -> list of sample_ids
    
    total_scenes = num_trainval_scenes + num_test_scenes
    class_names = ["sedan", "truck", "motorcycle", "emergency_vehicle"]
    
    sample_id_counter = 0
    
    for scene_idx in range(total_scenes):
        is_holdout_test = (scene_idx >= num_trainval_scenes)
        scene_id = f"scene_{scene_idx:02d}"
        
        for burst_idx in range(burst_per_scene):
            # Each burst is a distinct vehicle event passing through this scene
            cluster_id = f"event_s{scene_idx:02d}_b{burst_idx:02d}"
            ground_truth_clusters[cluster_id] = []
            
            # Target class for this burst event
            class_id = (scene_idx * 3 + burst_idx) % len(class_names)
            event_seed = (scene_idx * 1000) + (burst_idx * 50) + 7
            
            for cam_idx in range(num_cameras):
                cam_name = f"cam_{'front_left' if cam_idx == 0 else 'front_right'}"
                cam_angle = -1.2 if cam_idx == 0 else 1.2
                
                for frame_idx in range(frames_per_burst):
                    time_sec = float(frame_idx * 0.15) # 150ms apart
                    sample_id = f"img_{sample_id_counter:04d}"
                    sample_id_counter += 1
                    
                    # Generate visual content
                    var_type = "multi_camera" if cam_idx > 0 else "temporal_burst"
                    if frame_idx == 0 and cam_idx == 0:
                        var_type = "base"
                        
                    img = create_synthetic_object_image(
                        class_id=class_id,
                        base_seed=event_seed,
                        variation_type=var_type,
                        cam_angle_offset=cam_angle,
                        time_offset=time_sec
                    )
                    
                    filename = f"{sample_id}_{scene_id}_{cam_name}_f{frame_idx}.jpg"
                    filepath = os.path.join(images_dir, filename)
                    img.save(filepath, quality=92)
                    
                    record = {
                        "sample_id": sample_id,
                        "file_path": os.path.relpath(filepath, output_dir),
                        "scene_id": scene_id,
                        "camera_id": cam_name,
                        "burst_id": f"{scene_id}_b{burst_idx:02d}",
                        "frame_idx": frame_idx,
                        "timestamp_sec": round(burst_idx * 30.0 + time_sec, 2),
                        "class_id": class_id,
                        "class_name": class_names[class_id],
                        "cluster_id": cluster_id,
                        "is_holdout_scene": is_holdout_test
                    }
                    metadata.append(record)
                    ground_truth_clusters[cluster_id].append(sample_id)
                    
    # Generate pairwise ground truth leak pairs
    # Any pair belonging to the SAME cluster is a leak pair (if split across train/val)
    ground_truth_leak_pairs = []
    
    for cluster_id, sample_ids in ground_truth_clusters.items():
        n = len(sample_ids)
        for i in range(n):
            for j in range(i + 1, n):
                s1 = sample_ids[i]
                s2 = sample_ids[j]
                
                # Determine specific leak type:
                rec1 = next(m for m in metadata if m["sample_id"] == s1)
                rec2 = next(m for m in metadata if m["sample_id"] == s2)
                
                if rec1["camera_id"] == rec2["camera_id"]:
                    leak_type = "temporal_burst"
                else:
                    leak_type = "multi_camera_cross_view"
                    
                ground_truth_leak_pairs.append({
                    "sample_a": s1,
                    "sample_b": s2,
                    "cluster_id": cluster_id,
                    "leak_type": leak_type
                })
                
    # Save manifest and ground truth
    manifest_path = os.path.join(output_dir, "metadata.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    gt_pairs_path = os.path.join(output_dir, "ground_truth_leakage_pairs.json")
    with open(gt_pairs_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_samples": len(metadata),
            "total_clusters": len(ground_truth_clusters),
            "total_leak_pairs": len(ground_truth_leak_pairs),
            "leak_types": ["temporal_burst", "multi_camera_cross_view"],
            "pairs": ground_truth_leak_pairs
        }, f, indent=2)
        
    print(f"[DatasetGenerator] Generated {len(metadata)} samples ({num_trainval_scenes} train/val scenes, {num_test_scenes} holdout scenes).")
    print(f"[DatasetGenerator] Formed {len(ground_truth_clusters)} ground truth clusters with {len(ground_truth_leak_pairs)} pairwise leak relations.")
    print(f"[DatasetGenerator] Manifest saved to {manifest_path}")
    return manifest_path, gt_pairs_path

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "..", "dataset_store")
    generate_benchmark_dataset(out)
