import os
import subprocess
import whisper
from collections import defaultdict
import torch
from tqdm import tqdm


def merge_audio_files():
    files_dir = 'files'
    merged_dir = 'merged_files'
    os.makedirs(merged_dir, exist_ok=True)

    files_to_merge = defaultdict(list)
    lone_files = []
    for f in os.listdir(files_dir):
        if f.endswith('.m4a'):
            if '_' in f:
                base_name = f.split('_')[0]
                files_to_merge[base_name].append(os.path.join(files_dir, f))
            else:
                lone_files.append(os.path.join(files_dir, f))

    # Sort keys numerically when possible for a more natural order
    def sort_key(name):
        return (0, int(name)) if name.isdigit() else (1, name)

    print("Merging multipart audio files...")
    items = list(files_to_merge.items())
    items.sort(key=lambda kv: sort_key(kv[0]))
    for base_name, file_list in tqdm(items, desc="Merging", unit="file"):
        output_file = os.path.join(merged_dir, f"{base_name}.m4a")
        if os.path.exists(output_file):
            tqdm.write(f"Skipping merge for {base_name} (already exists)")
            continue
        tqdm.write(f"Merging {base_name} ({len(file_list)} parts)")
        list_file_path = os.path.join(merged_dir, f"{base_name}_list.txt")
        with open(list_file_path, 'w') as list_file:
            for file_path in sorted(file_list):
                list_file.write(f"file '{os.path.abspath(file_path)}'\n")
        subprocess.run([
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i',
            list_file_path, '-c', 'copy', output_file
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(list_file_path)

    print("Copying single audio files...")
    lone_files.sort(key=lambda p: sort_key(os.path.splitext(os.path.basename(p))[0]))
    for file_path in tqdm(lone_files, desc="Copying", unit="file"):
        output_file = os.path.join(merged_dir, os.path.basename(file_path))
        if os.path.exists(output_file):
            tqdm.write(f"Skipping copy for {os.path.basename(file_path)} (already exists)")
            continue
        tqdm.write(f"Copying {os.path.basename(file_path)}")
        subprocess.run(['cp', file_path, output_file], check=True)



def transcribe_audio():
    merged_dir = 'merged_files'
    transcripts_dir = 'transcripts'
    os.makedirs(transcripts_dir, exist_ok=True)

    device = "cpu"
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
            device = "mps"
            print("Using device: mps")
            model = whisper.load_model("base", device=device)
        except Exception as e:
            print(f"Failed to use MPS, falling back to CPU. Error: {e}")
            device = "cpu"
            model = whisper.load_model("base", device=device)
    else:
        print("Using device: cpu")
        model = whisper.load_model("base", device=device)

    files_to_transcribe = [f for f in os.listdir(merged_dir) if f.endswith('.m4a')]
    files_to_transcribe.sort(key=lambda name: (0, int(os.path.splitext(name)[0])) if os.path.splitext(name)[0].isdigit() else (1, name))

    print("\nTranscribing audio files...")
    for f in tqdm(files_to_transcribe, desc="Transcribing", unit="file"):
        transcript_path = os.path.join(transcripts_dir, f"{os.path.splitext(f)[0]}.txt")
        if os.path.exists(transcript_path):
            tqdm.write(f"Skipping transcription for {f} (already exists)")
            continue
        tqdm.write(f"Transcribing {f} on {device}")
        audio_path = os.path.join(merged_dir, f)
        result = model.transcribe(audio_path, verbose=False)
        with open(transcript_path, "w") as txt_file:
            txt_file.write(result["text"])


if __name__ == '__main__':
    merge_audio_files()
    transcribe_audio()
    print("\nAll files have been processed.") 