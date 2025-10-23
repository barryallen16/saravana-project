# counterfeit_detection_openrouter.py
# Batch processing for counterfeit bike parts detection using OpenRouter API
# Models: Qwen2.5-VL-32B or Gemini 2.0 Flash

import os
import json
import time
import base64
from tqdm import tqdm
import pandas as pd
import openai

# Configure OpenRouter API
client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    # api_key=os.getenv("OPENAI_API_KEY")
    
      api_key="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"  # Set via: export OPENROUTER_API_KEY=your_key
)

# ============================================================================
# EXPERT COUNTERFEIT DETECTION PROMPTS
# ============================================================================

COUNTERFEIT_DETECTION_PROMPTS = {
    'spark_plug': {
        'inspection_focus': [
            'Electrode alignment (straight vs bent/offset)',
            'Insulator ceramic color (white vs yellowed/discolored)',
            'Threading quality (crisp vs rough/stripped)',
            'Brand marking clarity (NGK/Denso/Champion - sharp vs blurry)',
            'Metal finish consistency (uniform vs scratchy/dull)',
            'Crimping area (smooth vs machine cutting marks)',
            'Hexagon section (LOT number present vs missing)',
            'C-groove area (smooth vs machining marks)',
            'Center electrode tip (iridium tip vs nickel)',
            'Ceramic-to-metal junction (white powder residue present)',
            'Paint color uniformity (consistent shade vs streaky)',
            'Overall weight appearance (substantial vs lightweight appearance)'
        ],
        'red_flags': [
            'Rotatable porcelain portion (indicates loose assembly)',
            'Machine cutting marks on crimping or C-groove (sign of remanufacturing)',
            'Blurry/smudged font on brand name',
            'Missing or incorrect LOT number on hexagon',
            'Electrode that appears tapered/sharpened artificially (lathe marks)',
            'Inconsistent paint color (darker or lighter areas)',
            'Poor ceramic-to-metal junction (rough interface)',
            'Visible machining marks on top rounded portion',
            'Missing white powder deposit at ceramic-metal junction (manufacturing residue)',
            'Incorrect or weak metal shell finish (shinier but messier than genuine)',
            'Heavy reflection inconsistencies on lettering (bumpy vs smooth)',
            'Lighter weight appearance compared to size'
        ]
    },
    'helmet': {
        'inspection_focus': [
            'Shell finish quality (smooth uniform vs rough/uneven paint)',
            'Paint consistency (uniform color vs patchy/streaky)',
            'Certification labels (ISI/DOT/ECE clear vs missing/blurry)',
            'Hologram security features (present vs missing)',
            'Serial number visibility (crisp emboss vs faded/missing)',
            'Strap stitching quality (tight uniform vs loose/wavy/skipped)',
            'Padding thickness (firm multi-layer vs single thin layer)',
            'Visor clarity (clear transparent vs cloudy/scratched/hazing)',
            'Visor seal integrity (snug vs loose fit)',
            'Ventilation holes (clean sharp edges vs rough/unfinished)',
            'Chin strap material (sturdy rigid vs flimsy plastic)',
            'Quick-release mechanism (smooth vs loose/missing)',
            'EPS foam thickness (visible ~1 inch vs thin/bare plastic)',
            'Base trim finishing (neat sealed edges vs rough/unfinished)',
            'Logo printing (sharp alignment vs misaligned/overlapping)',
            'D-ring attachment (secure rivets vs cheap plastic fasteners)'
        ],
        'red_flags': [
            'Missing ISI/DOT/ECE certification labels or hologram marks',
            'E-marking completely absent (especially on chin strap)',
            'Paint overlaps or misaligned seams (poor quality control)',
            'Single layer cheap foam padding instead of multi-density EPS',
            'Flimsy popper-only chin closure (no emergency quick-release tabs)',
            'Loose stitching, wavy seams, or skipped stitch patterns',
            'Cheap plastic chin strap instead of sturdy material',
            'Cloudy/scratched visor with haziness',
            'Uneven visor seal or gaps',
            'Poor logo printing (letters not aligned, wrong dimensions)',
            'Rough/unfinished ventilation holes',
            'Helmet weight significantly lighter than specifications (~1 pound vs 3 pounds)',
            'Missing or blurred serial numbers/date stamps',
            'Improperly applied stickers (peeling/bubbling)',
            'Removable liner that doesn\'t come out cleanly',
            'Base trim not evenly sealed'
        ]
    },
    'air_filter': {
        'inspection_focus': [
            'Pleating uniformity (consistent spacing vs irregular/uneven)',
            'Frame rigidity (sturdy vs flimsy/warped)',
            'Frame material (consistent thick material vs thin/bent)',
            'Seal/gasket integrity (complete coverage vs missing/partial)',
            'Seal material quality (flexible rubber vs hard/brittle)',
            'Filter media color (uniform vs patchy/faded)',
            'Filter media density (consistent thickness vs thin spots)',
            'Edge finishing (clean sealed vs frayed/rough edges)',
            'Edge seams (uniform welding vs loose/gaping)',
            'Brand logo clarity (sharp print vs blurry/faded)',
            'Mounting holes alignment (precise vs misaligned)',
            'Mounting hole finishing (smooth vs rough/burred)',
            'Overall frame finish (smooth vs rough/scratchy)',
            'Filter media pattern (uniform accordion vs wavy/collapsed sections)',
            'Weight/substance (feels substantial vs suspiciously light)',
            'Seal attachment (firmly bonded vs loose/peeling)'
        ],
        'red_flags': [
            'Uneven pleat spacing (irregular width, collapsed sections)',
            'Torn or damaged filter media (holes, punctures)',
            'Frayed/rough edges instead of sealed edges',
            'Deformed/warped frame (bent corners, twisted shape)',
            'Flimsy frame material (bends easily by hand)',
            'Missing seal or partial seal coverage (gaps visible)',
            'Hard brittle rubber seal that cracks easily (poor quality material)',
            'Seal peeling away from frame edges',
            'Filter media too thin (inadequate filtration depth)',
            'Blurry or faded brand logo/print',
            'Misaligned mounting holes (not symmetric)',
            'Burrs or rough edges on mounting holes',
            'Frame looks crushed or compressed at midpoint',
            'Weight suspiciously light (inadequate material content)',
            'Rough interior welds or seams (uneven bonding)',
            'Filter media separating from frame at edges'
        ]
    }
}

# ============================================================================
# PROMPT GENERATION
# ============================================================================

def generate_counterfeit_detection_prompt(part_type):
    """Generate detailed counterfeit detection prompt for specific part type"""
    
    data = COUNTERFEIT_DETECTION_PROMPTS.get(part_type, {})
    inspection = data.get('inspection_focus', [])
    red_flags = data.get('red_flags', [])
    
    # Format inspection points
    inspection_text = '\n'.join([f"  {i+1}. {point}" for i, point in enumerate(inspection[:8])])
    
    # Format red flags
    red_flags_text = '\n'.join([f"  • {flag}" for flag in red_flags[:10]])
    
    prompt = f"""COUNTERFEIT DETECTION ANALYSIS: {part_type.upper().replace('_', ' ')}

CONTEXT: This dataset contains ~50% genuine and ~50% counterfeit/defective {part_type.replace('_', ' ')}s. Genuine items show high manufacturing precision. Counterfeits have manufacturing defects.

INSPECTION AREAS TO EXAMINE:
{inspection_text}

MAJOR RED FLAGS FOR COUNTERFEITS:
{red_flags_text}

ANALYSIS FRAMEWORK:

Step 1: VISUAL ASSESSMENT
- Ignore text/labels - judge ONLY physical quality and manufacturing precision
- Compare overall build quality to professional standards
- Note any visible defects or imperfections

Step 2: MANUFACTURING QUALITY EVALUATION
Surface Defects (critical):
  - Rough, grainy, uneven surface texture
  - Visible scratches, scuffs, abrasions
  - Discoloration, staining, color inconsistencies
  - Corrosion, rust, oxidation, yellowing
  - Cracks, chips, breaks, material damage

Manufacturing Precision:
  - Misaligned parts, visible gaps, poor fit
  - Uneven spacing, uniformity problems
  - Loose/wavy stitching, skipped stitches
  - Poor edge finishing, frayed edges
  - Warped, bent, or deformed components
  - Imprecise mounting holes or attachments

Branding/Marking Defects:
  - Blurry, faded, or smudged text/logos
  - Incorrect font types or sizes
  - Misspellings or missing certifications
  - Inconsistent alignment or placement
  - Low-quality printing

Step 3: EVIDENCE WEIGHTING
- 3+ MAJOR defects → COUNTERFEIT (confidence 0.82-0.95)
- 2 MAJOR defects → COUNTERFEIT (confidence 0.68-0.82)
- 1 MAJOR defect OR multiple MINOR defects → COUNTERFEIT (confidence 0.60-0.72)
- Pristine condition, perfect precision, sharp details → GENUINE (confidence 0.80-0.95)
- Minor wear, good precision, clean details → GENUINE (confidence 0.70-0.80)
- Mixed signals, unclear → UNCERTAIN (confidence 0.45-0.60)

Step 4: PRIMARY REASONS (SPECIFIC OBSERVATIONS - not generic)
For GENUINE examples:
  - "Sharp electrode alignment with zero visible offset"
  - "Crisp white ceramic insulator with no discoloration"
  - "Perfectly uniform pleating with consistent spacing"
  - "Sturdy frame with no visible warping or deformation"

For COUNTERFEIT examples:
  - "Rough surface with visible machine cutting marks on crimping area"
  - "Blurred/smudged NGK logo indicating poor print quality"
  - "Collapsed/irregular pleating with wavy sections and tone variation"
  - "Flimsy frame warped at multiple points, bends easily by hand"
  - "Frayed edges throughout perimeter, poor seal integrity"

OUTPUT: Return ONLY valid JSON (no other text):
{{
  "classification": "GENUINE" or "COUNTERFEIT",
  "confidence": 0.85,
  "primary_reasons": [
    "specific observation 1",
    "specific observation 2",
    "specific observation 3"
  ]
}}

BE CRITICAL: Counterfeits have obvious manufacturing flaws. Report what you SEE, not what you assume. If image quality is poor, set confidence below 0.60."""
    
    return prompt

# ============================================================================
# OPENROUTER API CLASSIFICATION
# ============================================================================

def classify_bike_part_openrouter(image_path, part_type, model="qwen/qwen2.5-vl-32b-instruct:free"):
    """
    Classify bike part as GENUINE or COUNTERFEIT using OpenRouter API
    
    Args:
        image_path: Path to image file
        part_type: 'spark_plug', 'helmet', or 'air_filter'
        model: OpenRouter model identifier
    
    Returns:
        dict with classification, confidence, and reasons
    """
    
    try:
        # Read and encode image as base64
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Determine image type
        ext = os.path.splitext(image_path)[1].lower()
        media_type = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.webp': 'image/webp'
        }.get(ext, 'image/jpeg')
        
        # Generate part-specific prompt
        prompt = generate_counterfeit_detection_prompt(part_type)
        
        # Call OpenRouter API
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_data}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }],
            temperature=0.3,
            max_tokens=512,
            top_p=0.85
        )
        
        output_text = response.choices[0].message.content
        
        # Parse JSON response
        json_start = output_text.find('{')
        json_end = output_text.rfind('}') + 1
        
        if json_start == -1 or json_end <= 0:
            return None
        
        json_str = output_text[json_start:json_end]
        result = json.loads(json_str)
        
        # Validate output
        if result.get("classification") not in ["GENUINE", "COUNTERFEIT"]:
            return None
        
        result['confidence'] = float(result.get('confidence', 0.5))
        
        return result
        
    except json.JSONDecodeError as e:
        print(f"⚠️  JSON parse error: {os.path.basename(image_path)}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# ============================================================================
# BATCH PROCESSING WITH RATE LIMITING
# ============================================================================

def validate_dataset_batch_openrouter(data_list, existing_jsonl, output_jsonl, 
                                      start=None, end=None, rate_limit_delay=6.0,
                                      model="qwen/qwen2.5-vl-32b-instruct:free"):
    """
    Run batch VLM validation on dataset with OpenRouter API
    
    Args:
        data_list: List of image data dictionaries with 'image_path', 'part_type', 'claimed_label'
        existing_jsonl: Path to main results file (for merging batches)
        output_jsonl: Output file for this batch
        start: Starting index (0-based inclusive)
        end: Ending index (exclusive)
        rate_limit_delay: Seconds between API calls (6+ for free tier safety)
        model: OpenRouter model to use
    """
    
    processed_paths = set()
    
    # Set defaults
    if start is None:
        start = 0
    
    # Validate range
    if start < 0:
        start = 0
    if end is not None and end > len(data_list):
        end = len(data_list)
    if end is not None and start >= end:
        print(f"❌ Error: start ({start}) must be < end ({end})")
        return
    
    # Load existing results from main file
    if os.path.exists(existing_jsonl):
        print(f"📂 Loading existing: {existing_jsonl}")
        with open(existing_jsonl, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    processed_paths.add(item['image_path'])
                except:
                    continue
        print(f"✅ Found {len(processed_paths)} previously processed\n")
    
    # Also check output file for partial progress
    if os.path.exists(output_jsonl):
        with open(output_jsonl, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    processed_paths.add(item['image_path'])
                except:
                    continue
    
    # Slice range
    if end is not None:
        selected_items = data_list[start:end]
        range_desc = f"{start} to {end-1}"
        total_in_range = end - start
    else:
        selected_items = data_list[start:]
        range_desc = f"{start} to end"
        total_in_range = len(selected_items)
    
    # Filter unprocessed
    items_to_process = [
        item for item in selected_items
        if item['image_path'] not in processed_paths
    ]
    
    skipped = len(selected_items) - len(items_to_process)
    
    # Print status
    print("="*80)
    print("BATCH VALIDATION - OPENROUTER API")
    print("="*80)
    print(f"📌 Range: items {range_desc} ({total_in_range} images)")
    print(f"📊 Already processed: {skipped}")
    print(f"📊 New to process: {len(items_to_process)}")
    print(f"🤖 Model: {model}")
    print(f"⏱️  Rate limit: {rate_limit_delay}s between calls")
    print(f"📁 Output: {output_jsonl}")
    
    if len(items_to_process) > 0:
        est_time = len(items_to_process) * rate_limit_delay / 60
        print(f"⏳ Estimated time: {est_time:.1f} minutes\n")
    else:
        print("✅ All items already processed!")
        return
    print("="*80 + "\n")
    
    # Process with progress bar and rate limiting
    start_time = time.time()
    
    for idx, item in enumerate(tqdm(items_to_process, desc="Validating", unit="img")):
        # Call API
        result = classify_bike_part_openrouter(
            item['image_path'],
            item['part_type'],
            model=model
        )
        
        if result:
            # Compare with folder label
            vlm_says_fake = (result['classification'] == 'COUNTERFEIT')
            folder_says_fake = (item['claimed_label'] == 'fake')
            agrees = (vlm_says_fake == folder_says_fake)
            
            entry = {
                'filename': item['filename'],
                'image_path': item['image_path'],
                'part_type': item['part_type'],
                'folder_name': item.get('folder_name', ''),
                'claimed_label': item['claimed_label'],
                'vlm_classification': result['classification'],
                'vlm_confidence': result['confidence'],
                'vlm_reasons': result.get('primary_reasons', []),
                'agreement': agrees,
                'needs_review': (not agrees) or (result['confidence'] < 0.65)
            }
            
            # Append to file (crash recovery)
            with open(output_jsonl, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        
        # Rate limiting (respect free tier limits)
        if idx < len(items_to_process) - 1:
            time.sleep(rate_limit_delay)
    
    elapsed = time.time() - start_time
    
    # Summary
    print(f"\n{'='*80}")
    print(f"✅ BATCH COMPLETE")
    print(f"{'='*80}")
    print(f"✅ Range: {range_desc}")
    print(f"✅ Processed: {len(items_to_process)} images")
    print(f"✅ Time: {elapsed/60:.1f} minutes ({elapsed/len(items_to_process):.1f}s per image)")
    print(f"✅ Output: {output_jsonl}")
    print(f"{'='*80}\n")

# ============================================================================
# ANALYSIS AND REPORTING
# ============================================================================

def analyze_validation_results(jsonl_path):
    """Analyze and report validation results"""
    
    print("="*80)
    print("VALIDATION ANALYSIS")
    print("="*80)
    
    df = pd.read_json(jsonl_path, lines=True)
    total = len(df)
    
    agreements = df['agreement'].sum()
    disagreements = total - agreements
    
    print(f"\n📊 Overall Performance:")
    print(f"   Total validated: {total}")
    print(f"   Agreements: {agreements} ({agreements/total*100:.1f}%)")
    print(f"   Disagreements: {disagreements} ({disagreements/total*100:.1f}%)")
    
    # Confidence distribution
    print(f"\n📈 Confidence Distribution:")
    high_conf = (df['vlm_confidence'] >= 0.80).sum()
    med_conf = ((df['vlm_confidence'] >= 0.65) & (df['vlm_confidence'] < 0.80)).sum()
    low_conf = (df['vlm_confidence'] < 0.65).sum()
    
    print(f"   High (≥0.80): {high_conf} ({high_conf/total*100:.1f}%)")
    print(f"   Medium (0.65-0.80): {med_conf} ({med_conf/total*100:.1f}%)")
    print(f"   Low (<0.65): {low_conf} ({low_conf/total*100:.1f}%)")
    
    # By part type
    print(f"\n🔧 Agreement by Part Type:")
    for part in sorted(df['part_type'].unique()):
        part_df = df[df['part_type'] == part]
        part_agree = part_df['agreement'].sum()
        part_total = len(part_df)
        print(f"   {part:15s}: {part_agree}/{part_total} ({part_agree/part_total*100:.1f}%)")
    
    # Priority review
    print(f"\n⚠️  PRIORITY REVIEW:")
    
    high_disagree = df[(df['agreement'] == False) & (df['vlm_confidence'] >= 0.80)]
    print(f"   High-conf disagreements: {len(high_disagree)}")
    
    low_conf_all = df[df['vlm_confidence'] < 0.65]
    print(f"   Low-confidence: {len(low_conf_all)}")
    
    total_review = df['needs_review'].sum()
    print(f"   Total need review: {total_review} ({total_review/total*100:.1f}%)\n")
    
    # Export review files
    if len(high_disagree) > 0:
        high_disagree[['filename', 'part_type', 'claimed_label', 'vlm_classification', 
                       'vlm_confidence']].to_csv('review_high_priority.csv', index=False)
        print(f"   ✅ Exported: review_high_priority.csv")
    
    if total_review > 0:
        review_df = df[df['needs_review'] == True]
        review_df[['filename', 'part_type', 'claimed_label', 'vlm_classification',
                   'vlm_confidence']].to_csv('review_needed.csv', index=False)
        print(f"   ✅ Exported: review_needed.csv")
    
    print("="*80 + "\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Load data_list
    with open('input/data_list.json', 'r') as f:
        data_list = json.load(f)
    
    # ========== CONFIGURATION ==========
    
    # Choose model
    MODEL = "qwen/qwen2.5-vl-32b-instruct:free"  # Best for counterfeit detection
    # ALT: "google/gemini-2.0-flash-exp:free"    # Alternative (slightly lower accuracy)
    
    # Choose batch range
    START = 150
    END = 200  # First 50 images
    # For multiple Colab: 0-150, 150-300, 300-444
    
    RATE_LIMIT_DELAY = 6.0  # Seconds (safety margin for free tier)
    
    # ========== RUN BATCH ==========
    
    validate_dataset_batch_openrouter(
        data_list=data_list,
        existing_jsonl="validation_results.jsonl",
        output_jsonl=f"output/batch_{START}_{END}.jsonl",
        start=START,
        end=END,
        rate_limit_delay=RATE_LIMIT_DELAY,
        model=MODEL
    )
    
    # ========== MERGE & ANALYZE (after all batches complete) ==========
    
    # Uncomment after running all batches:
    
    # print("\n" + "="*80)
    # print("MERGING ALL BATCHES")
    # print("="*80)
    # 
    # batch_files = [
    #     'batch_0_150.jsonl',
    #     'batch_150_300.jsonl',
    #     'batch_300_444.jsonl'
    # ]
    # 
    # with open('validation_results.jsonl', 'w') as outfile:
    #     for batch_file in batch_files:
    #         if os.path.exists(batch_file):
    #             with open(batch_file, 'r') as infile:
    #                 for line in infile:
    #                     outfile.write(line)
    #             print(f"✅ Merged: {batch_file}")
    # 
    # print(f"\n✅ All batches merged!\n")
    # analyze_validation_results("validation_results.jsonl")
