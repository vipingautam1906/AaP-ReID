import os
import sys


def read_file_contents(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
    return lines

def get_log_train_files(directory):
    log_train_files = []
    for root, _, files in os.walk(directory):
        for file_name in files:
            if file_name == "log_train.txt":
                log_train_files.append(os.path.join(root, file_name))
    return log_train_files

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python list_log_train_files.py <directory_path>")
        sys.exit(1)

    directory_path = sys.argv[1]
    
    log_train_files = get_log_train_files(directory_path)
    if not log_train_files:
        print("No 'log_train.txt' files found in the specified directory.")
    else:
        for file_path in log_train_files:
            lines = read_file_contents(file_path)
            for line in lines:
                if 'Epoch' in line:
                    print(line.strip())

