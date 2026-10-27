'''
Feb/6/2026 - Fri

Testing - not yet apply the minio, mongo 
Test for 1 image

'''

import torch
import yaml
import numpy as np
import os
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToTensor

# nam@NamLaptop:~/Downloads/source_for_seq2seq_manual_label_5Feb2026$ python3 -m prelabeller_ocr.inference.OCR_Yolo_Seq2Seq

'''
Updated process_image_crop: It now strictly follows your logic: Resize (keeping aspect ratio) --> ToTensor --> No Normalization.
==> for Seq2Seq
'''

# --- Imports from your project structure ---
# Ensure this script is placed where it can access these modules
from ..postprocessing.NonMaxSurpression import non_max_suppression, Yolo2Pix
from ..models.ocr_builder import (
    Seq2Seq__Independ_VietOCR,
    translate,
    process_image_crop,
    build_ocr_models_from_cfg,
)

def visualize_results(image_path, results, output_dir):
    """
    Draws bounding boxes and text on the original image.
    results: list of (x1, y1, x2, y2, text)
    """
    try:
        img = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        # # Try to load a font, otherwise default
        # try:
        #     # Linux usually has this font. Adjust size as needed.
        #     font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
        # except IOError:
        #     font = ImageFont.load_default()

        for res in results:
            x1, y1, x2, y2 = res
            
            # Draw Box (Green)
            draw.rectangle([x1, y1, x2, y2], outline="green", width=1)
            
            # Draw Text Background (Green) and Text (White)
            # Calculate text size using getbbox (left, top, right, bottom)
            # text_bbox = draw.textbbox((x1, y1), text, font=font)
            # text_bbox = draw.textbbox((x1, y1))

            # text_w = text_bbox[2] - text_bbox[0]
            # text_h = text_bbox[3] - text_bbox[1]
            
            # draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill="green")
            # draw.text((x1 + 2, y1 - text_h - 4), text, fill="white", font=font)

        # Save visualization
        file_name = Path(image_path).stem + "_visualized.jpg"
        save_path = Path(output_dir) / file_name
        img.save(save_path)
        print(f"Visualization saved to: {save_path}")
        
    except Exception as e:
        print(f"Error during visualization: {e}")


# --- YOLO Helper Functions ---
def clip_boxes(boxes, shape):
    if isinstance(boxes, torch.Tensor):
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    ratio, pad = ratio_pad
    pad_x, pad_y = pad
    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y
    boxes[:, :4] /= ratio
    clip_boxes(boxes, img0_shape)
    return boxes

def Letterbox_resize(image, target_img_size=(480, 480)):
    original_width, original_height = image.size  
    ratio = min(target_img_size[0] / original_width, target_img_size[1] / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)    

    new_image = Image.new('RGB', target_img_size, (0, 0, 0))
    padding_x = (target_img_size[0] - new_width) // 2
    padding_y = (target_img_size[1] - new_height) // 2
    new_image.paste(resized_image, (padding_x, padding_y))

    return new_image, ratio, (padding_x, padding_y)

# ==========================================
# 2. Main Pipeline
# ==========================================

def run_ocr_pipeline(config_path):
    # --- Load Config ---
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    print(f"Running pipeline on: {cfg['device']}")

    # ============================
    # A. LOAD MODELS
    # ============================
    print("Loading OCR models (YOLO11 + Seq2Seq)...")
    try:
        ocr_models = build_ocr_models_from_cfg(cfg)
    except Exception as e:
        print(f"Error building OCR models: {e}")
        return

    yolo_model = ocr_models.yolo_model
    rec_model = ocr_models.rec_model
    vocab = ocr_models.vocab
    device = ocr_models.device

    print(f"Vocab length in YAML {len(cfg['seq2seq']['vocab'])}")
    print(f"Vocab Length after add 4 special token LLM: {len(vocab)}")


    # ============================
    # B. INFERENCE
    # ============================
    
    img_path = cfg['input_image_path']
    print(f"Processing: {img_path}")
    
    # 1. Preprocess for Detection
    img0 = Image.open(img_path).convert('RGB')
    img0_np = np.array(img0)
    target_size = (cfg['yolo']['input_size'][0], cfg['yolo']['input_size'][1])
    
    img_resized, ratio, padding_values = Letterbox_resize(img0, target_img_size=target_size)
    
    # Prepare YOLO Tensor
    img_np = np.array(img_resized)
    img_np = img_np.transpose((2, 0, 1)) # HWC -> CHW
    img_np = np.ascontiguousarray(img_np)
    img_tensor_yolo = torch.from_numpy(img_np).to(device).float()
    img_tensor_yolo /= 255.0
    if len(img_tensor_yolo.shape) == 3:
        img_tensor_yolo = img_tensor_yolo.unsqueeze(0)

    # 2. Run Detection
    with torch.no_grad():
        pred = yolo_model(img_tensor_yolo)

    # 3. Post-process Detection
    box_predictions = Yolo2Pix(pred[..., :4])
    other_predictions = pred[..., 4:]
    pred_pixel_coords = torch.cat((box_predictions, other_predictions), dim=-1)

    pred_nms = non_max_suppression(
        pred_pixel_coords, 
        confidence_threshold=cfg['yolo']['conf_thres'], 
        iou_threshold=cfg['yolo']['iou_thres']
    )

    # ============================
    # C. CROP & RECOGNIZE & SAVE
    # ============================
    
    output_lines = []
    visualization_data = [] # Store data for visualization
    
    for i, det in enumerate(pred_nms):
        if len(det):
            # Scale boxes back to original image size
            det[:, :4] = scale_boxes(
                img_resized.size,
                det[:, :4], 
                img0_np.shape, 
                ratio_pad=(ratio, padding_values)
            )

            # Sort boxes
            det = det[det[:, 1].argsort()]

            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                
                # --- Recognition Step ---
                # Crop from original PIL image
                crop_img = img0.crop((x1, y1, x2, y2))
                
                # Prepare crop for Seq2Seq (Process EXACTLY as requested)
                rec_tensor = process_image_crop(
                    crop_img, 
                    cfg['seq2seq']['height'], 
                    device
                )
                
                # Predict Text
                predicted_text = translate(rec_tensor, rec_model, vocab, device)
                
                # --- Store Data ---
                # 1. For TXT File
                line_str = f"{x1},{y1},{x2},{y1},{x2},{y2},{x1},{y2},{predicted_text}"
                output_lines.append(line_str)
                
                # 2. For Visualization
                visualization_data.append((x1, y1, x2, y2))
                
                print(f"Detected: {predicted_text} at [{x1}, {y1}, {x2}, {y2}]")

    # ============================
    # D. WRITE TO FILES
    # ============================
    
    out_dir = Path(cfg['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save Text Label
    file_name = Path(img_path).stem + ".txt"
    out_path = out_dir / file_name
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
        
    print(f"------------------------------------------------")
    print(f"Saved text result to: {out_path}")
    
    # 2. Save Visualization
    visualize_results(img_path, visualization_data, out_dir)
    print(f"------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    BASE_DIR = Path(__file__).resolve().parent.parent 
    DEFAULT_CONFIG_PATH = BASE_DIR / "config" /"OCR_Yolo_Seq2Seq_config.yaml"
    parser.add_argument('--config', type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to config file")
    args = parser.parse_args()
    
    run_ocr_pipeline(args.config)