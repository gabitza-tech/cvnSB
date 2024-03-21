import os
import csv
import sys
import json

# Function to traverse directories and write file paths to CSV
def write_wav_paths_to_csv(root_dir, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=' ')
        csvwriter.writerow(['audio_name_without_extension', 'audio_path'])
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.wav'):
                    audio_name = os.path.splitext(file)[0]
                    audio_path = os.path.join(subdir, file)
                    csvwriter.writerow([audio_name, audio_path])


def write_manifest_to_csv(root_directory,csv_file):
    with open(csv_file, mode='w', newline='') as file, open(root_directory, mode='r') as f:

        lines = f.readlines()
        writer = csv.writer(file, delimiter= ' ')
        # Write data
        for line in lines:
            item = json.loads(line)
            writer.writerow([item["file_id"], item["audio_filepath"]])

# Specify your root directory containing subdirectories with .wav files
root_directory = sys.argv[1]

# Specify the output CSV file path
output_csv_file = sys.argv[2]

# Call the function
if os.path.isdir(root_directory):
    write_wav_paths_to_csv(root_directory, output_csv_file)
else:
    write_manifest_to_csv(root_directory,output_csv_file)