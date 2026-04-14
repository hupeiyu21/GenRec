import argparse
import gzip
import os
import shutil
import sys
import tarfile
import zipfile
from typing import Dict

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util import Retry
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download dataset by dataset from url.yaml and extract to datasets folder."
    )
    parser.add_argument("--dataset", required=True, help="dataset defined in url.yaml")
    parser.add_argument(
        "--url_file",
        default=os.path.join(os.path.dirname(__file__), "url.yaml"),
        help="Path to url.yaml",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join("..", "datasets"),
        help="Directory to save downloaded and extracted datasets",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extracted folder if it already exists",
    )
    return parser.parse_args()


def build_target_dir(output_root: str, dataset_name: str) -> str:
    return os.path.join(output_root, f"recbole_{dataset_name}")


def load_url_map(url_file: str) -> Dict[str, str]:
    if not os.path.exists(url_file):
        raise FileNotFoundError(f"URL config not found: {url_file}")
    with open(url_file, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def build_session(retries: int = 5, backoff: float = 1.0) -> requests.Session:
    """Create a requests session with retry to mitigate transient SSL/connection errors."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        status=retries,
        backoff_factor=backoff,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def download_with_progress(url: str, dest_path: str):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        print(f"[Info] File already exists, skip download: {dest_path}")
        return

    session = build_session()
    try:
        with session.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            bar = tqdm(total=total, unit="iB", unit_scale=True, desc="Downloading")
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        bar.update(len(chunk))
                        f.write(chunk)
            bar.close()
        print(f"[Info] Downloaded to {dest_path}")
    except Exception:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise


def _safe_join(base: str, *paths: str) -> str:
    final_path = os.path.realpath(os.path.join(base, *paths))
    if not final_path.startswith(os.path.realpath(base) + os.sep):
        raise RuntimeError(f"Unsafe path detected: {final_path}")
    return final_path


def _replace_path(src_path: str, dst_path: str):
    if os.path.isdir(dst_path):
        shutil.rmtree(dst_path)
    elif os.path.exists(dst_path):
        os.remove(dst_path)
    shutil.move(src_path, dst_path)


def _collect_extracted_roots(extract_dir: str, members):
    roots = []
    for member in members:
        if not member:
            continue
        roots.append(os.path.join(extract_dir, member.split("/")[0]))
    return [os.path.abspath(path) for path in roots if os.path.exists(path)]


def extract_and_rename_all(archive_path: str, dataset_name: str, target_dir: str):
    base_dir = os.path.dirname(os.path.abspath(archive_path))
    extract_dir = os.path.join(base_dir, f".extract_{dataset_name}")
    lower = archive_path.lower()
    extracted_roots = []

    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)

    try:
        if lower.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zf:
                members = zf.namelist()
                for member in members:
                    _safe_join(extract_dir, member)
                zf.extractall(extract_dir)
                extracted_roots = _collect_extracted_roots(extract_dir, members)

        elif lower.endswith((".tar.gz", ".tgz", ".tar")):
            mode = "r:gz" if lower.endswith((".tar.gz", ".tgz")) else "r"
            with tarfile.open(archive_path, mode) as tf:
                members = [member.name for member in tf.getmembers()]
                for member in members:
                    _safe_join(extract_dir, member)
                tf.extractall(extract_dir)
                extracted_roots = _collect_extracted_roots(extract_dir, members)

        elif lower.endswith(".gz"):
            out = os.path.join(extract_dir, os.path.basename(archive_path)[:-3])
            with gzip.open(archive_path, "rb") as fin, open(out, "wb") as fout:
                shutil.copyfileobj(fin, fout)
            extracted_roots = [out]

        else:
            return

        extracted_roots = list(dict.fromkeys(extracted_roots))
        if not extracted_roots:
            raise RuntimeError(f"No files extracted from archive: {archive_path}")

        if len(extracted_roots) == 1 and os.path.isdir(extracted_roots[0]):
            _replace_path(extracted_roots[0], target_dir)
        else:
            os.makedirs(target_dir, exist_ok=True)
            for root in extracted_roots:
                dest_path = os.path.join(target_dir, os.path.basename(root))
                if os.path.abspath(root) != os.path.abspath(dest_path):
                    _replace_path(root, dest_path)

        for dirpath, _, filenames in os.walk(target_dir):
            for fname in filenames:
                old_path = os.path.join(dirpath, fname)
                _, ext = os.path.splitext(fname)
                new_path = os.path.join(dirpath, f"{dataset_name}{ext}")
                if old_path != new_path:
                    if os.path.exists(new_path):
                        os.remove(new_path)
                    os.rename(old_path, new_path)
    finally:
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)


def main():
    args = parse_args()
    try:
        url_map = load_url_map(args.url_file)
    except Exception as e:
        print(f"[Error] Failed to load url map: {e}")
        sys.exit(1)

    if args.dataset not in url_map:
        print(f"[Error] dataset '{args.dataset}' not found in {args.url_file}")
        print("Available datasets (partial):")
        for key in list(url_map.keys())[:20]:
            print(f"  - {key}")
        sys.exit(1)

    url = url_map[args.dataset]
    output_root = os.path.abspath(args.output_dir)
    os.makedirs(output_root, exist_ok=True)

    archive_name = os.path.basename(url)
    archive_path = os.path.join(output_root, archive_name)
    target_dir = build_target_dir(output_root, args.dataset)

    if os.path.exists(target_dir) and os.listdir(target_dir) and not args.overwrite:
        print(f"[Info] Target directory already exists and not empty: {target_dir}")
        print("       Use --overwrite to force re-download and extraction.")
        sys.exit(0)
    if os.path.exists(target_dir) and args.overwrite:
        shutil.rmtree(target_dir)

    try:
        download_with_progress(url, archive_path)
        extract_and_rename_all(archive_path, args.dataset, target_dir)
        if os.path.exists(archive_path):
            os.remove(archive_path)
            print(f"[Info] Removed archive: {archive_path}")
        print(f"[Done] Dataset '{args.dataset}' is ready at: {target_dir}")
    except Exception as e:
        print(f"[Error] Failed to download or extract: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
