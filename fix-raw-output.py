import json
import re

def extract_json_from_raw_output(raw_text):
    """
    Aggressively extract JSON from raw model output.
    Handles cases where model generates lots of reasoning text.
    """
    
    # Remove </think> and everything before it
    if '</think>' in raw_text:
        raw_text = raw_text[raw_text.find('</think>') + 8:]
    
    # Try to find JSON object
    json_start = raw_text.find('{')
    if json_start == -1:
        return None
    
    # Find matching closing brace
    brace_count = 0
    json_end = -1
    
    for i in range(json_start, len(raw_text)):
        if raw_text[i] == '{':
            brace_count += 1
        elif raw_text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                json_end = i + 1
                break
    
    if json_end == -1:
        return None
    
    json_str = raw_text[json_start:json_end]
    
    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError:
        return None


def process_raw_outputs_file(input_file, output_file, failed_file):
    """
    Process raw_outputs.jsonl and attempt to extract JSON from all entries.
    Saves successfully parsed results and logs failures.
    """
    
    successful = 0
    failed = 0
    
    failed_entries = []
    successful_entries = []
    
    print("Processing raw_outputs.jsonl...")
    print("="*80)
    
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())
                
                # Extract JSON from raw_output
                parsed_json = extract_json_from_raw_output(entry['raw_output'])
                
                if parsed_json:
                    # Validate JSON structure
                    if 'classification' in parsed_json and 'confidence' in parsed_json:
                        # Add metadata
                        result_entry = {
                            'image_path': entry['image_path'],
                            'filename': entry['filename'],
                            'part_type': entry['part_type'],
                            'vlm_classification': parsed_json.get('classification'),
                            'vlm_confidence': float(parsed_json.get('confidence', 0)),
                            'vlm_reasons': parsed_json.get('primary_reasons', []),
                            'timestamp': entry.get('timestamp')
                        }
                        successful_entries.append(result_entry)
                        successful += 1
                    else:
                        failed_entries.append({
                            'image_path': entry['image_path'],
                            'filename': entry['filename'],
                            'reason': 'Missing required JSON fields',
                            'raw_output_preview': entry['raw_output'][:300]
                        })
                        failed += 1
                else:
                    failed_entries.append({
                        'image_path': entry['image_path'],
                        'filename': entry['filename'],
                        'reason': 'Could not extract valid JSON',
                        'raw_output_preview': entry['raw_output'][:300]
                    })
                    failed += 1
                    
            except json.JSONDecodeError as e:
                failed += 1
                failed_entries.append({
                    'line': line_num,
                    'reason': f'Invalid JSON in raw_outputs: {e}'
                })
            except Exception as e:
                failed += 1
                failed_entries.append({
                    'line': line_num,
                    'reason': f'Error processing: {e}'
                })
    
    # Write successful results
    print(f"\n✅ Successfully parsed: {successful}")
    with open(output_file, 'w') as f:
        for entry in successful_entries:
            f.write(json.dumps(entry) + '\n')
    print(f"   Saved to: {output_file}")
    
    # Write failed results for review
    print(f"\n⚠️  Failed to parse: {failed}")
    with open(failed_file, 'w') as f:
        for entry in failed_entries:
            f.write(json.dumps(entry) + '\n')
    print(f"   Saved to: {failed_file}")
    
    print("\n" + "="*80)
    print(f"Success rate: {successful}/{successful+failed} ({100*successful/(successful+failed):.1f}%)")
    print("="*80)

# Run the processing
process_raw_outputs_file(
    input_file='raw_outputs.jsonl',
    output_file='parsed_results.jsonl',
    failed_file='parsing_failures.jsonl'
)