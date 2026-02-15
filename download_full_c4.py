import argparse
import os
import datasets

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Download and cache the C4 dataset.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="~/data",
        help="The directory where you want to cache the dataset. Defaults to '~/data'.",
    )
    return parser.parse_args()

def main():
    """
    Downloads the 'train' and 'validation' splits of the 'allenai/c4' (en) dataset
    to a specified cache directory.
    """
    args = parse_args()
    
    cache_dir = os.path.join(os.path.expanduser(args.data_dir), '')
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Starting to download the C4 dataset to: {cache_dir}")
    print("This might take a significant amount of time and disk space.")
    
    print(f"HF_HOME: {os.environ.get('HF_HOME', 'Not set')}")
    print(f"HF_DATASETS_CACHE: {os.environ.get('HF_DATASETS_CACHE', 'Not set')}")
    print(f"Cache directory: {cache_dir}")

    print(f"Starting to download the C4 dataset to: {cache_dir}")
    print("This might take a significant amount of time and disk space.")

    try:
        # Download the training split
        print("\nDownloading the 'train' split...")
        datasets.load_dataset(
            "allenai/c4",
            "en",
            split="train",
            streaming=False,
            cache_dir=cache_dir,
            num_proc=4,
        )
        print("Successfully downloaded the 'train' split.")

        # Download the validation split
        print("\nDownloading the 'validation' split...")
        datasets.load_dataset(
            "allenai/c4",
            "en",
            split="validation",
            streaming=False,
            cache_dir=cache_dir,
            num_proc=4,
        )
        print("Successfully downloaded the 'validation' split.")

        print(f"\nDataset download complete. Cached at: {cache_dir}")

    except Exception as e:
        print(f"\nAn error occurred during download: {e}")

if __name__ == "__main__":
    main()