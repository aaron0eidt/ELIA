import os
import sys
from pathlib import Path
import requests
import json
import time
from tqdm import tqdm

# Add root project dir to path
sys.path.append(str(Path(__file__).parent.parent))
from function_vectors.data.multilingual_function_categories import FUNCTION_CATEGORIES, FUNCTION_TYPES

# API configuration for Qwen.
QWEN_API_CONFIG = {
    "api_key": os.environ.get("QWEN_API_KEY", "YOUR_API_KEY_HERE"),
    "api_endpoint": "https://chat-ai.academiccloud.de/v1",
    "model": "qwen2.5-vl-72b-instruct",
    "rate_limit_per_minute": 2,
}

# --- Translation Logic ---

def translate_text(text, target_language="German"):
    # Translates a single string using the Qwen API.
    headers = {
        "Authorization": f"Bearer {QWEN_API_CONFIG['api_key']}",
        "Content-Type": "application/json"
    }
    
    prompt = f"Translate the following English text to {target_language}. Respond with ONLY the translated text, without any introductory phrases, explanations, or quotation marks. The original text is:\n\n'{text}'"

    data = {
        "model": QWEN_API_CONFIG["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
        "temperature": 0.1,
    }
    
    try:
        response = requests.post(
            f"{QWEN_API_CONFIG['api_endpoint']}/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            translated_text = result["choices"][0]["message"]["content"].strip()
            # Clean up quotes from the model's response.
            if translated_text.startswith('"') and translated_text.endswith('"'):
                translated_text = translated_text[1:-1]
            return translated_text
        elif response.status_code == 429:
            # Handle rate limiting.
            reset_time = response.headers.get('RateLimit-Reset', '0')
            try:
                wait_seconds = int(reset_time)
                print(f"Hourly rate limit reached. Waiting {wait_seconds} seconds for reset...")
                return f"RATE_LIMIT_HOURLY:{wait_seconds}"
            except ValueError:
                print("Rate limit exceeded. Waiting 60 seconds...")
                return "RATE_LIMIT_EXCEEDED"
        else:
            print(f"API Error: Status {response.status_code}, Response: {response.text}")
            return None
            
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None

def translate_batch_texts(texts, target_language="German"):
    # Translates a batch of strings in one API call.
    headers = {
        "Authorization": f"Bearer {QWEN_API_CONFIG['api_key']}",
        "Content-Type": "application/json"
    }
    # A stronger prompt to ensure full translation.
    batch_prompt = (
        f"Translate the following English texts to {target_language}. "
        "For each text, translate ALL words and phrases, including any words in quotation marks, into natural German. "
        "Do NOT leave any English words in the translation. Respond with ONLY the German translations, one per line, in the same order.\n\n"
    )
    for i, text in enumerate(texts, 1):
        batch_prompt += f"{i}. {text}\n"
    batch_prompt += "\nProvide the German translations in the same order, one per line:"
    data = {
        "model": QWEN_API_CONFIG["model"],
        "messages": [{"role": "user", "content": batch_prompt}],
        "max_tokens": 300,  # Increased for batch processing
        "temperature": 0.1,
    }
    try:
        response = requests.post(
            f"{QWEN_API_CONFIG['api_endpoint']}/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )
        if response.status_code == 200:
            result = response.json()
            translated_text = result["choices"][0]["message"]["content"].strip()
            # Split the response into individual lines.
            lines = [line.strip() for line in translated_text.split('\n') if line.strip()]
            cleaned_translations = []
            for line in lines:
                # Remove numbering if the model adds it.
                if line and line[0].isdigit() and '.' in line:
                    line = line.split('.', 1)[1].strip()
                # Clean up quotes.
                if line.startswith('"') and line.endswith('"'):
                    line = line[1:-1]
                if line:
                    cleaned_translations.append(line)
            # Make sure we have the right number of translations.
            if len(cleaned_translations) >= len(texts):
                return cleaned_translations[:len(texts)]
            else:
                print(f"Warning: Expected {len(texts)} translations, got {len(cleaned_translations)}")
                # Pad with error messages if some translations failed.
                while len(cleaned_translations) < len(texts):
                    cleaned_translations.append(f"TRANSLATION_ERROR: {texts[len(cleaned_translations)]}")
                return cleaned_translations
        elif response.status_code == 429:
            # Handle rate limiting.
            reset_time = response.headers.get('RateLimit-Reset', '0')
            try:
                wait_seconds = int(reset_time)
                print(f"Hourly rate limit reached. Waiting {wait_seconds} seconds for reset...")
                return f"RATE_LIMIT_HOURLY:{wait_seconds}"
            except ValueError:
                print("Rate limit exceeded. Waiting 60 seconds...")
                return "RATE_LIMIT_EXCEEDED"
        else:
            print(f"API Error: Status {response.status_code}, Response: {response.text}")
            return None
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None

def update_multilingual_categories_file(new_categories):
    # Updates the multilingual_function_categories.py file.
    file_path = Path(__file__).parent / "data" / "multilingual_function_categories.py"
    
    # Create the new file content.
    file_content = "# -*- coding: utf-8 -*-\n"
    file_content += '"""\nThis file contains the multilingual prompts for function vector analysis.\n'
    file_content += 'It is automatically updated by the translate_prompts.py script.\n"""\n\n'
    
    # Format the FUNCTION_TYPES dictionary.
    ft_content = "FUNCTION_TYPES = {\n"
    for ft, cats in FUNCTION_TYPES.items():
        ft_content += f'    "{ft}": [\n'
        for cat in cats:
            ft_content += f'        "{cat}",\n'
        ft_content += "    ],\n"
    ft_content += "}\n\n"

    file_content += ft_content

    # Add the function categories.
    file_content += f"FUNCTION_CATEGORIES = {json.dumps(new_categories, indent=4, ensure_ascii=False)}\n"
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(file_content)
    print(f"\n✅ Progress saved to '{file_path}'")


def main():
    # Translates all prompts and updates the file.
    print("🚀 Starting batch translation of prompts to German...")

    # Load existing categories to resume from where we left off.
    translated_categories = FUNCTION_CATEGORIES.copy()
    
    # Count how many prompts need to be translated.
    total_prompts = sum(len(prompts.get('en', [])) for prompts in FUNCTION_CATEGORIES.values())
    
    # Set up a progress bar.
    with tqdm(total=total_prompts, desc="Translating Prompts") as pbar:
        # Check how many are already translated.
        already_translated_count = 0
        for category_key, data in FUNCTION_CATEGORIES.items():
            if 'de' not in translated_categories.get(category_key, {}):
                if category_key not in translated_categories:
                    translated_categories[category_key] = {}
                translated_categories[category_key]['de'] = []

            if 'de' in translated_categories[category_key]:
                 already_translated_count += len(translated_categories[category_key]['de'])
        pbar.update(already_translated_count)

        # Get a list of all prompts that still need to be translated.
        all_prompts_to_translate = []
        prompt_mapping = []
        
        for category_key, data in FUNCTION_CATEGORIES.items():
            english_prompts = data.get('en', [])
            
            # Make sure the 'de' key exists.
            if 'de' not in translated_categories[category_key]:
                translated_categories[category_key]['de'] = []
            
            german_prompts = translated_categories[category_key]['de']

            # Skip if this category is already done.
            if len(german_prompts) == len(english_prompts):
                continue

            # Add prompts that are missing a translation.
            for i in range(len(german_prompts), len(english_prompts)):
                all_prompts_to_translate.append(english_prompts[i])
                prompt_mapping.append((category_key, i))

        # Process the prompts in batches.
        batch_size = 6
        for i in range(0, len(all_prompts_to_translate), batch_size):
            batch_prompts = all_prompts_to_translate[i:i + batch_size]
            batch_mapping = prompt_mapping[i:i + batch_size]
            
            # Wait between batches to avoid hitting the rate limit.
            time.sleep(30)
            
            translated_batch = translate_batch_texts(batch_prompts)
            
            # Handle rate limit responses.
            if translated_batch and isinstance(translated_batch, str) and translated_batch.startswith("RATE_LIMIT_HOURLY:"):
                wait_seconds = int(translated_batch.split(":")[1])
                print(f"Waiting {wait_seconds} seconds for hourly rate limit reset...")
                time.sleep(wait_seconds)
                # Retry the batch.
                translated_batch = translate_batch_texts(batch_prompts)
            
            retry_wait = 60
            while translated_batch == "RATE_LIMIT_EXCEEDED":
                # Wait and retry if we hit the rate limit.
                print(f"Waiting for {retry_wait} seconds due to rate limit...")
                time.sleep(retry_wait)
                translated_batch = translate_batch_texts(batch_prompts)
                retry_wait *= 1.5
            
            if translated_batch and isinstance(translated_batch, list):
                # Add the new translations to our data.
                for j, (category_key, prompt_idx) in enumerate(batch_mapping):
                    if j < len(translated_batch):
                        translated_categories[category_key]['de'].append(translated_batch[j])
                
                # Save progress every so often.
                if (pbar.n + len(batch_prompts)) % 30 == 0:
                    update_multilingual_categories_file(translated_categories)
                
                pbar.update(len(batch_prompts))
            else:
                print(f"❌ Failed to translate batch. Stopping.")
                # Save any progress we made before stopping.
                update_multilingual_categories_file(translated_categories)
                return

    # Final save at the end.
    update_multilingual_categories_file(translated_categories)
    print("\n✅ All prompts translated and file updated successfully.")

if __name__ == "__main__":
    main() 