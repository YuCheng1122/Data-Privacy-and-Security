import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# ---------- CONFIG ----------
ORIGINAL_DIR = "./celeba_original"
BLURRED_DIR = "./blurred_images_50"
OUTPUT_DIR = "./dp_blurred_images_50_eps_5"
FILENAME_LIST = "./test_filenames.txt"  # Can be changed to .csv if needed
EPSILON = 5
# ----------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

def apply_dp_noise(image_array, epsilon):
    image = image_array.astype(np.float32) / 255.0
    sigma = 1.0 / epsilon
    noise = np.random.normal(0, sigma, image.shape)
    noisy = np.clip(image + noise, 0, 1)
    return (noisy * 255).astype(np.uint8)

def calculate_metrics(img1, img2):
    ssim_score = ssim(img1, img2, channel_axis=-1)
    mse_score = mean_squared_error(img1.flatten(), img2.flatten())
    return ssim_score, mse_score

def load_filename_list(path):
    if path.endswith(".txt"):
        with open(path, "r") as f:
            return [line.strip() for line in f.readlines()]
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
        return df["filename"].tolist() if "filename" in df.columns else df.iloc[:, 0].tolist()
    else:
        raise ValueError("Unsupported filename list format. Use .txt or .csv.")

def main():
    filenames_to_process = load_filename_list(FILENAME_LIST)
    done_filenames = set(os.listdir(OUTPUT_DIR))
    results = []

    print(f"Processing {len(filenames_to_process)} specified images...")

    for filename in tqdm(filenames_to_process, desc="Processing", unit="img"):
        if filename in done_filenames:
            continue

        blurred_path = os.path.join(BLURRED_DIR, filename)
        original_path = os.path.join(ORIGINAL_DIR, filename)

        if not os.path.exists(blurred_path) or not os.path.exists(original_path):
            continue

        blurred_img = np.array(Image.open(blurred_path).convert("RGB"))
        original_img = np.array(Image.open(original_path).convert("RGB"))

        dp_img = apply_dp_noise(blurred_img, EPSILON)
        Image.fromarray(dp_img).save(os.path.join(OUTPUT_DIR, filename))

        ssim_score, mse_score = calculate_metrics(original_img, dp_img)
        results.append({
            "filename": filename,
            "ssim": ssim_score,
            "mse": mse_score
        })

    df = pd.DataFrame(results)
    df.to_csv(f"dp_results_eps_{EPSILON}.csv", index=False)

    print(f"\nFinished processing {len(results)} images with DP. Results saved to: dp_results_eps_{EPSILON}.csv")

if __name__ == "__main__":
    main()
