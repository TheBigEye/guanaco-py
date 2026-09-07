"""Optional explicit GGUF downloader (requires huggingface-hub); never runs during a build."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", help="Hugging Face repository ID")
    parser.add_argument("filename", help="Explicit .gguf file to download")
    parser.add_argument(
        "--revision", default="main", help="Use an immutable revision for reproducibility"
    )
    parser.add_argument("--directory", type=Path, default=Path("models"))
    args = parser.parse_args()
    if not args.filename.lower().endswith(".gguf"):
        parser.error("Use a GGUF model; obsolete GGML .bin files are not supported")
    from huggingface_hub import hf_hub_download

    print(
        hf_hub_download(
            repo_id=args.repository,
            filename=args.filename,
            revision=args.revision,
            local_dir=str(args.directory),
        )
    )


if __name__ == "__main__":
    main()
