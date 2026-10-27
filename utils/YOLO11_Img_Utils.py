from PIL import Image
import torch

'''
Temporary "Fixed" size for image - 10/10/2025
'''

# Resize letter box
def Letterbox_resize(image, target_img_size=(480, 480)):
    
    # Read image size
    original_width, original_height = image.size  
    ratio = min(target_img_size[0] / original_width, target_img_size[1] / original_height) # Resize Ratio
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)    

    # Create a new image with padding
    new_image = Image.new('RGB', target_img_size, (0, 0, 0))
    padding_x = (target_img_size[0] - new_width) // 2
    padding_y = (target_img_size[1] - new_height) // 2
    new_image.paste(resized_image, (padding_x, padding_y))

    return new_image, ratio, (padding_x, padding_y)


# Scale box for Inference 
def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """
    Rescales bounding boxes (xyxy) from the resized image shape to the original image shape.
    """
    ratio, pad = ratio_pad
    # gain = ratio  # FIXED: ratio is already the min ratio (float)
    pad_x, pad_y = pad

    boxes[:, [0, 2]] -= pad_x  # x padding
    boxes[:, [1, 3]] -= pad_y  # y padding
    boxes[:, :4] /= ratio
    clip_boxes(boxes, img0_shape)
    return boxes


# Trim and cut edge 
def clip_boxes(boxes, shape):
    """
    Clips = cut off or trim something to a specific boundary. Clips the bounding box (xyxy) coordinates to fit within the image shape (height, width).
    This is a safety measure to prevent out-of-bounds coordinates after scaling.
    """
    if isinstance(boxes, torch.Tensor):
        boxes[:, 0].clamp_(0, shape[1])  # x1 (width)
        boxes[:, 1].clamp_(0, shape[0])  # y1 (height)
        boxes[:, 2].clamp_(0, shape[1])  # x2 (width)
        boxes[:, 3].clamp_(0, shape[0])  # y2 (height)
    else: # numpy
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])