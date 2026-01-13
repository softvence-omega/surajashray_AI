from app.utils.pinecone_utils import upsert_document
import json
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple
import os

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, str]:
    """Flatten a nested dictionary into a single level dictionary."""
    items: List[Tuple[str, str]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, (list, tuple)):
            # Handle lists by joining elements
            items.append((new_key, ', '.join(map(str, v))))
        else:
            items.append((new_key, str(v)))
    
    return dict(items)

def format_value(key: str, value: Any) -> str:
    """Format a value based on common patterns in the key name."""
    if isinstance(value, (int, float)):
        if any(unit in key.lower() for unit in ['temp', 'temperature']):
            return f"{value}°C"
        elif any(unit in key.lower() for unit in ['weight', 'mass']):
            return f"{value}kg"
        elif any(unit in key.lower() for unit in ['height', 'length']):
            return f"{value}cm"
        elif 'time' in key.lower() or 'duration' in key.lower():
            return f"{value}min"
    return str(value)

def detect_data_type(data: Dict[str, Any]) -> str:
    """Detect the type of data based on keys and structure."""
    key_hints = {
        'lab': ['test', 'result', 'value', 'reference', 'range'],
        'profile': ['name', 'age', 'gender', 'height', 'weight'],
        'vital': ['heart_rate', 'blood_pressure', 'temperature'],
        'sleep': ['duration', 'quality', 'stages'],
        'nutrition': ['calories', 'meal', 'nutrients'],
        'activity': ['steps', 'exercise', 'workout'],
        'medical': ['diagnosis', 'prescription', 'treatment']
    }
    
    flat_data = flatten_dict(data)
    flat_keys = ' '.join(flat_data.keys()).lower()
    
    for data_type, hints in key_hints.items():
        if any(hint in flat_keys for hint in hints):
            return data_type
    
    return 'general'

def format_data_entry(data: Dict[str, Any], prefix: str = '') -> List[Tuple[str, str]]:
    """Format a data entry into a list of (id, text) tuples."""
    formatted_entries = []
    flat_data = flatten_dict(data)
    data_type = detect_data_type(data)
    timestamp = datetime.now().isoformat()
    
    # Group related fields together
    if data_type == 'lab':
        # Format lab results
        test_name = flat_data.get('test_name', 'Unknown Test')
        value = flat_data.get('value', 'N/A')
        unit = flat_data.get('unit', '')
        ref_range = flat_data.get('reference_range', flat_data.get('ref_ranges', 'N/A'))
        status = flat_data.get('status', 'N/A')
        
        text = f"Lab test {test_name}: value {value} {unit} (reference range: {ref_range}, status: {status})"
        formatted_entries.append((f"{prefix}lab_{test_name.lower().replace(' ', '_')}", text))
    
    elif data_type == 'profile':
        # Format profile information
        profile_fields = {k: v for k, v in flat_data.items() if any(field in k.lower() for field in ['name', 'age', 'gender', 'height', 'weight'])}
        if profile_fields:
            text = "User Profile - " + ", ".join(f"{k.split('_')[-1].title()}: {format_value(k, v)}" for k, v in profile_fields.items())
            formatted_entries.append((f"{prefix}profile", text))
    
    else:
        # General formatting for other types of data
        text = f"{data_type.title()} Data - " + ", ".join(f"{k.split('_')[-1].title()}: {format_value(k, v)}" for k, v in flat_data.items())
        formatted_entries.append((f"{prefix}{data_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}", text))
    
    return formatted_entries

def process_json_file(file_path: str) -> List[Tuple[str, str]]:
    """Process a JSON file and return formatted entries."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    formatted_entries = []
    
    def process_dict(d: Dict[str, Any], prefix: str = '') -> None:
        for key, value in d.items():
            if isinstance(value, dict):
                # Process nested dictionaries
                formatted_entries.extend(format_data_entry(value, f"{prefix}{key}_"))
                process_dict(value, f"{prefix}{key}_")
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                # Process lists of dictionaries
                for i, item in enumerate(value):
                    formatted_entries.extend(format_data_entry(item, f"{prefix}{key}_{i}_"))
    
    process_dict(data)
    return formatted_entries

def main():
    parser = argparse.ArgumentParser(description='Process JSON files and insert into Pinecone vector database')
    parser.add_argument('json_files', nargs='+', help='Path to JSON file(s) to process')
    args = parser.parse_args()
    
    for json_file in args.json_files:
        if not os.path.exists(json_file):
            print(f"Error: File {json_file} not found")
            continue
            
        print(f"Processing {json_file}...")
        try:
            docs = process_json_file(json_file)
            for doc_id, text in docs:
                print(f"Inserting {doc_id}...")
                result = upsert_document(doc_id, text)
                print(result)
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")

if __name__ == "__main__":
    main()
