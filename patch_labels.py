import cv2
import numpy as np
from PIL import Image
import os

def get_label_box(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_box = None
    max_score = 0
    center_x = img.shape[1] / 2
    center_y = img.shape[0] / 2
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area > (img.shape[0] * img.shape[1]) * 0.05:
            dist = abs((x + w/2) - center_x) + abs((y + h/2) - center_y)
            score = area / (dist + 1)
            if score > max_score:
                max_score = score
                best_box = (x, y, w, h)
    return best_box

def patch(image_path, lockup_path, output_path):
    box = get_label_box(image_path)
    if not box:
        print(f"Failed to find label in {image_path}")
        return
    x, y, w, h = box
    print(f"Patching {image_path} label at {box}")
    
    img_cv = cv2.imread(image_path)
    # Get median color of the label
    label_roi = img_cv[y:y+h, x:x+w]
    med = np.median(label_roi, axis=(0,1))
    bg_color = (int(med[0]), int(med[1]), int(med[2]))
    
    # Fill inner part of the label to preserve borders
    shrink = 25
    cv2.rectangle(img_cv, (x+shrink, y+shrink), (x+w-shrink, y+h-shrink), bg_color, -1)
    
    cv2.imwrite(output_path, img_cv)
    
    # Now paste lockup
    img = Image.open(output_path).convert("RGBA")
    logo = Image.open(lockup_path).convert("RGBA")
    
    target_w = int(w * 0.75)
    target_h = int(target_w * logo.height / logo.width)
    if target_h > h * 0.75:
        target_h = int(h * 0.75)
        target_w = int(target_h * logo.width / logo.height)
        
    paste_x = x + (w - target_w) // 2
    paste_y = y + (h - target_h) // 2
    
    resized_logo = logo.resize((target_w, target_h), Image.Resampling.LANCZOS)
    img.paste(resized_logo, (paste_x, paste_y), resized_logo)
    img.convert('RGB').save(output_path, "PNG")
    print(f"Saved {output_path}")

lockup_path = '/tmp/lockups/lockup_black.png'
brain_beef = '/Users/alex/.gemini/antigravity/brain/8ae195f2-8ec9-4a75-a237-2b6fd6c482f9/gallery_packaged_braised_beef_1773438036876.png'
site_beef = '/Users/alex/src/xijiade/gallery_packaged_braised_beef.png'
site_chicken = '/Users/alex/src/xijiade/hero_packaged_braised_chicken.png'

patch(brain_beef, lockup_path, site_beef)
patch(site_chicken, lockup_path, site_chicken)
