import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# ---- Differential Privacy Methods ----

def dp_pix(image, block_size, epsilon, m):
    h, w = image.shape[:2]
    output = np.zeros_like(image, dtype=np.float32)
    sensitivity = 255 * m / (block_size ** 2)
    scale = sensitivity / epsilon

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            roi = image[y:y + block_size, x:x + block_size]
            mean_val = roi.mean(axis=(0, 1), keepdims=True)
            noise = np.random.laplace(0, scale, size=(1, 1, 3))
            noisy_val = np.clip(mean_val + noise, 0, 255)
            output[y:y + block_size, x:x + block_size] = noisy_val

    return output.astype(np.uint8)

def dp_blur(image, block_size, epsilon, m, kernel_size):
    dp_pix_img = dp_pix(image, block_size, epsilon, m)
    upsampled = cv2.resize(dp_pix_img, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    blurred = cv2.GaussianBlur(upsampled, (kernel_size, kernel_size), 0)
    return blurred

# ---- Metric Computation ----

def compute_metrics(original, obfuscated):
    mse_val = np.mean((original - obfuscated) ** 2)
    ssim_val = ssim(original, obfuscated, channel_axis=-1)
    return mse_val, ssim_val

# ---- Image Processing ----

def process_image(img_path, out_dir, fname, method='origin', block_size=16, epsilon=0.5, m=16, kernel_size=45):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))

    if method == 'dp_pix':
        obf = dp_pix(img, block_size, epsilon, m)
    elif method == 'dp_blur':
        obf = dp_blur(img, block_size, epsilon, m, kernel_size)
    elif method == 'origin':
        obf = img.copy()
    else:
        raise ValueError(f"Unsupported method: {method}")

    if method != 'origin':
        # out_label_dir = os.path.join(out_dir, method)
        # os.makedirs(out_label_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, fname), obf)
        
    return img, obf

# ---- Main Execution ----

if __name__ == '__main__':
    input_dir = './HW3/hw3/celeba_original'
    output_dir = './HW3/dp_output'

    methods = ['dp_pix', 'dp_blur']
    config = {
        'dp_pix': {
            'block_size': 16,
            'epsilon': 0.2,
            'm': 16
        },
        'dp_blur': {
            'block_size': 16,
            'epsilon': 0.2,
            'm': 16,
            'kernel_size': 45
        }
    }

    max_images = None 

    all_images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if max_images:
        all_images = all_images[:max_images]

    for method in methods:
        print(f"\n[INFO] Processing method: {method}")
        output_method_dir = os.path.join(output_dir, method)
        os.makedirs(output_method_dir, exist_ok=True)

        for fname in all_images:
            img_path = os.path.join(input_dir, fname)
            output_path = os.path.join(output_method_dir, fname)

            if not os.path.exists(img_path):
                print(f"[SKIP] File not found: {img_path}")
                continue

            try:
                _, obf = process_image(
                    img_path,
                    output_method_dir,
                    fname,
                    method=method,
                    block_size=config[method]['block_size'],
                    epsilon=config[method]['epsilon'],
                    m=config[method]['m'],
                    kernel_size=config[method].get('kernel_size', 45)
                )
                print(f"[OK] Saved: {output_path}")
            except Exception as e:
                print(f"[ERROR] Failed processing {fname}: {e}")