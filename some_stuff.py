import os
import shutil

def separate_images_by_class(jpeg_folder, class_txt_path, output_dir, is_test=False):
    # Create output directories
    class_name = os.path.basename(class_txt_path).split('_')[0]
    in_class_dir = os.path.join(output_dir, f"{class_name}")
    not_class_dir = os.path.join(output_dir, f"not_{class_name}")

    if not is_test:
        os.makedirs(in_class_dir, exist_ok=True)
        os.makedirs(not_class_dir, exist_ok=True)
    # Read labels
    with open(class_txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue  # skip malformed lines
            image_id, label = parts
            if not is_test:
                label = int(label)
                src_path = os.path.join(jpeg_folder, f"{image_id}.jpg")

                if not os.path.exists(src_path):
                    print(f"Warning: {src_path} not found.")
                    continue

                if label == 1:
                    dst_path = os.path.join(in_class_dir, f"{image_id}.jpg")
                elif label == -1:
                    dst_path = os.path.join(not_class_dir, f"{image_id}.jpg")
                else:
                    # Ignore ambiguous (label 0)
                    continue

                shutil.copy2(src_path, dst_path)
            else:
                # For test images, we assume they are all in the same folder
                src_path = os.path.join(jpeg_folder, f"{image_id}.jpg")
                if not os.path.exists(src_path):
                    print(f"Warning: {src_path} not found.")
                    continue
                dst_path = os.path.join(output_dir, f"{image_id}.jpg")
                shutil.copy2(src_path, dst_path)

    print(f"✅ Done. Images copied to:\n  {in_class_dir}\n  {not_class_dir}")

separate_images_by_class("nndl_proj/JPEGImages_test", "nndl_proj/meta_datas/bird_test.txt", "nndl_proj/data/test_set", is_test=True)