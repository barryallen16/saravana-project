# create_clean_dataset.py
import os
import json
import shutil
import pandas as pd

def create_clean_dataset(validation_jsonl, corrections_csv, output_dir='dataset_clean'):
    """
    Create cleaned dataset with corrected labels
    
    Folder structure:
    dataset_clean/
      train/
        spark_plug_genuine/
        spark_plug_fake/
        helmet_genuine/
        helmet_fake/
        air_filter_genuine/
        air_filter_fake/
      valid/
        (same structure)
    """
    
    # Load validation results
    with open(validation_jsonl, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Load human corrections if exists
    corrections = {}
    if os.path.exists(corrections_csv):
        corr_df = pd.read_csv(corrections_csv)
        corrections = {row['image_path']: row['human_correction'] 
                       for _, row in corr_df.iterrows()}
        print(f"✅ Loaded {len(corrections)} human corrections")
    
    # Create output directories
    for split in ['train', 'valid']:
        for part in ['spark_plug', 'helmet', 'air_filter']:
            for label in ['genuine', 'fake']:
                os.makedirs(f"{output_dir}/{split}/{part}_{label}", exist_ok=True)
    
    # Process each image
    genuine_count = 0
    fake_count = 0
    corrected_count = 0
    
    for idx, item in enumerate(data):
        image_path = item['image_path']
        
        # Determine final label
        if image_path in corrections:
            # Use human correction
            final_label = corrections[image_path]
            corrected_count += 1
        elif item['agreement']:
            # Use folder label (VLM agreed)
            final_label = item['claimed_label']
        else:
            # Use VLM prediction (disagreement, no human review yet)
            # OPTION 1: Trust VLM (recommended if 88% agreement)
            final_label = 'fake' if item['vlm_classification'] == 'COUNTERFEIT' else 'real'
            # OPTION 2: Keep original (conservative)
            # final_label = item['claimed_label']
        
        # Convert label
        label_folder = 'genuine' if final_label == 'real' else 'fake'
        
        # Train/valid split (80/20)
        split = 'train' if idx % 5 != 0 else 'valid'
        
        # Copy to new location
        part = item['part_type']
        dest_folder = f"{output_dir}/{split}/{part}_{label_folder}"
        dest_path = os.path.join(dest_folder, item['filename'])
        
        shutil.copy2(image_path, dest_path)
        
        if label_folder == 'genuine':
            genuine_count += 1
        else:
            fake_count += 1
    
    # Print summary
    print("\n" + "="*80)
    print("CLEAN DATASET CREATED")
    print("="*80)
    print(f"Output directory: {output_dir}/")
    print(f"Total images: {len(data)}")
    print(f"Human corrections applied: {corrected_count}")
    print(f"Genuine: {genuine_count} ({genuine_count/len(data)*100:.1f}%)")
    print(f"Fake: {fake_count} ({fake_count/len(data)*100:.1f}%)")
    
    # Print split details
    for split in ['train', 'valid']:
        total_split = sum(1 for i, x in enumerate(data) if (i % 5 != 0 if split == 'train' else i % 5 == 0))
        print(f"\n{split.upper()} split: {total_split} images (~{total_split/len(data)*100:.0f}%)")
        
        for part in ['spark_plug', 'helmet', 'air_filter']:
            genuine_path = f"{output_dir}/{split}/{part}_genuine"
            fake_path = f"{output_dir}/{split}/{part}_fake"
            genuine_n = len(os.listdir(genuine_path))
            fake_n = len(os.listdir(fake_path))
            print(f"  {part:15s}: genuine={genuine_n:3d}, fake={fake_n:3d}")
    
    print("="*80 + "\n")

# Create clean dataset
create_clean_dataset(
    'validation_results.jsonl',
    'human_corrections.csv',  # Will use if exists
    'dataset_clean'
)
