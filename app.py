import os
import random
import json
from collections import Counter

data_list = []

# Your actual folder structure
folders = [
    'spark plug fake',
    'spark plug og',
    'helmet fake', 
    'helmet og',
    'air filter fake',  # Fixed: should be 'fake' not 'real'
    'air filter og'
]

print("="*80)
print("LOADING DATASET")
print("="*80)

for folder in folders:
    folder_path = f"dataset1/{folder}"
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"⚠️  Warning: Folder not found: {folder_path}")
        continue
    
    # Determine part type
    if 'spark' in folder:
        part_type = 'spark_plug'
    elif 'helmet' in folder:
        part_type = 'helmet'
    elif 'filter' in folder:
        part_type = 'air_filter'
    else:
        print(f"⚠️  Unknown part type in folder: {folder}")
        continue
    
    # Determine label: 'og' = genuine/real, 'fake' = counterfeit
    claimed_label = 'real' if 'og' in folder else 'fake'
    
    # Load images
    image_count = 0
    for image_file in os.listdir(folder_path):
        # Skip non-image files
        if not image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
            continue
        
        data_list.append({
            'image_path': os.path.join(folder_path, image_file),
            'part_type': part_type,
            'claimed_label': claimed_label,
            'folder_name': folder,
            'filename': image_file
        })
        image_count += 1
    
    print(f"✅ {folder:25s} → {image_count:4d} images (label: {claimed_label})")

# Shuffle to remove bias
random.seed(42)  # For reproducibility
random.shuffle(data_list)

print(f"\n{'='*80}")
print(f"DATASET SUMMARY")
print(f"{'='*80}")
print(f"Total images: {len(data_list)}")

# Count by part type and label
part_counts = Counter([item['part_type'] for item in data_list])
label_counts = Counter([item['claimed_label'] for item in data_list])

print(f"\n📦 By Part Type:")
for part, count in sorted(part_counts.items()):
    fake_count = len([x for x in data_list if x['part_type'] == part and x['claimed_label'] == 'fake'])
    real_count = len([x for x in data_list if x['part_type'] == part and x['claimed_label'] == 'real'])
    print(f"  {part:15s}: {count:4d} total (real: {real_count:3d}, fake: {fake_count:3d})")

print(f"\n🏷️  By Label:")
for label, count in sorted(label_counts.items()):
    print(f"  {label:15s}: {count:4d} ({count/len(data_list)*100:.1f}%)")

# Check class balance
if 'fake' in label_counts and 'real' in label_counts:
    fake_count = label_counts['fake']
    real_count = label_counts['real']
    balance_ratio = min(fake_count, real_count) / max(fake_count, real_count)
    
    print(f"\n⚖️  Class Balance: {balance_ratio:.1%}")
    if balance_ratio < 0.7:
        print(f"   ⚠️  Imbalanced! Consider data augmentation for minority class")
    elif balance_ratio >= 0.9:
        print(f"   ✅ Well balanced!")
    else:
        print(f"   ✓  Acceptable balance")

print(f"{'='*80}\n")

# Save data list for later use
with open('data_list.json', 'w') as f:
    json.dump(data_list, f, indent=2)
print("💾 Saved data_list to 'data_list.json'\n")

# Show sample entries
print("Sample entries:")
for i, item in enumerate(data_list[:3]):
    print(f"{i+1}. {item['filename'][:40]:40s} | {item['part_type']:12s} | {item['claimed_label']}")
