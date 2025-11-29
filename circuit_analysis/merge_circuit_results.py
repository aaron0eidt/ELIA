import json
import sys
import os

def merge_json_results(base_file, new_file):
    # Merges the 'analyses' from a new results file into a base file.
    try:
        # Ensure the results directory exists.
        results_dir = "circuit_analysis/results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        base_path = os.path.join(results_dir, base_file)
        new_path = os.path.join(results_dir, new_file)

        # Load the base file, or create a new one if it doesn't exist.
        if os.path.exists(base_path):
            with open(base_path, 'r') as f:
                base_data = json.load(f)
        else:
            print(f"Base file '{base_file}' not found. Creating a new one.")
            base_data = {"analyses": {}}

        # Load the new results file.
        with open(new_path, 'r') as f:
            new_data = json.load(f)
            
        # Ensure both files have the 'analyses' key.
        if 'analyses' not in base_data or 'analyses' not in new_data:
            print("Error: Both files must contain an 'analyses' key.")
            return

        # Update the analyses from the base file.
        base_data['analyses'].update(new_data['analyses'])
        
        # Update the timestamp and config from the new file.
        base_data['timestamp'] = new_data.get('timestamp', base_data.get('timestamp'))
        base_data['config'] = new_data.get('config', base_data.get('config'))

        # Write the merged data back to the base file.
        with open(base_path, 'w') as f:
            json.dump(base_data, f, indent=2)
            
        print(f"Successfully merged '{new_file}' into '{base_file}'.")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e.filename}")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in one of the files.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python merge_circuit_results.py <base_json_file> <new_json_file>")
    else:
        merge_json_results(sys.argv[1], sys.argv[2]) 