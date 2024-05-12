import os
import json
import argparse
import fsspec
import braceexpand
import concurrent.futures as cf
import multiprocessing as mp
from rich.progress import Progress
from rich.console import Console
from dataclasses import dataclass, field


@dataclass
class State:
    shards: set[str] = field(default_factory=set)

    def to_json(self) -> str:
        data = {
            "shards": list(self.shards),
        }
        return json.dumps(data)


def wds_init(fs: fsspec.AbstractFileSystem):
    import functools
    import boto3
    import boto3.session
    from webdataset import gopen_schemes
    from urllib.parse import urlparse

    @functools.cache
    def build_s3_client():
        creds_file = os.path.expanduser("~/.config/fsspec/s3.json")
        with open(creds_file) as f:
            config = json.load(f)
        config = config["s3"]["client_kwargs"]
        s3_client = boto3.client(
            service_name="s3",
            endpoint_url=config["endpoint_url"],
            aws_access_key_id=config["aws_access_key_id"],
            aws_secret_access_key=config["aws_secret_access_key"],
            config=boto3.session.Config(
                retries={"max_attempts": 10, "mode": "standard"}
            ),
        )
        return s3_client

    def gopen_s3_python(url, mode="rb", bufsize=8192):
        """Open a URL with an S3 client.

        :param url: url (s3://)
        """
        storage_client = build_s3_client()

        parsed_url = urlparse(url)
        if parsed_url.scheme != "s3":
            raise ValueError("gopen_s3_python only works with s3:// urls")

        bucket_name = parsed_url.netloc
        blob_name = parsed_url.path.removeprefix("/")

        data = storage_client.get_object(Bucket=bucket_name, Key=blob_name)
        return data["Body"]

    gopen_schemes["s3"] = gopen_s3_python


def exists_fn(
    fs: fsspec.AbstractFileSystem,
    wds_file: str,
    metadata_file: str,
) -> None:
    import webdataset as wds

    wds_init(fs)
    dataset = (
        wds.WebDataset(fs.unstrip_protocol(wds_file))
        .decode("pil", handler=wds.ignore_and_continue)
        .to_tuple("__key__", "jpg", "json")
        .select(lambda items: items[1].size >= (256, 256))
        .batched(batchsize=32)
    )

    num_samples = 0
    for keys, jpgs, jsons in dataset:
        num_samples += len(keys)

    print(f"Found {num_samples} samples in {wds_file}.")


def caption_fn(
    fs: fsspec.AbstractFileSystem,
    wds_file: str,
    metadata_file: str,
) -> None:
    pass


def run_event_loop(
    console: Console,
    action: str,
    state: State,
    fs: fsspec.AbstractFileSystem,
    wds_paths: list[str],
    metadata_dir: str,
    num_workers: int,
):
    # Primary event loop for managing the execution.
    if num_workers == 1:
        executor = cf.ThreadPoolExecutor(max_workers=num_workers)
    else:
        executor = cf.ProcessPoolExecutor( # type: ignore
            max_workers=num_workers,
            mp_context=mp.get_context("spawn"),
        )

    if action == "exists":
        action_fn = exists_fn
    elif action == "caption":
        action_fn = caption_fn
    else:
        raise ValueError(f"Unknown action: {action}")

    with executor, Progress() as progress:
        task = progress.add_task("Processing...", total=len(wds_paths))
        futures: set[cf.Future[None]] = set()
        while wds_paths or futures:
            for _ in range(num_workers - len(futures)):
                if not wds_paths:
                    break

                fs_path = wds_paths.pop()
                metadata_file = os.path.join(metadata_dir, os.path.basename(fs_path))

                future = executor.submit(
                    action_fn,
                    fs,
                    fs_path,
                    metadata_file,
                )
                future.add_done_callback(lambda _: progress.update(task, advance=1))
                futures.add(future)

            done, futures = cf.wait(futures, return_when=cf.FIRST_COMPLETED)
            for future in done:
                try:
                    future.result()
                except Exception as e:
                    console.print(f"Error processing {fs_path}: {e}")
                else:
                    state.shards.add(fs_path)
                    fs.write_text(
                        os.path.join(metadata_dir, "state.json"),
                        state.to_json(),
                    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the dataset."
    )
    parser.add_argument(
        "--metadata-dir",
        type=str,
        required=True,
        help="Path to the metadata store. Must be in the same filesystem as the dataset.",
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["exists", "caption"],
        required=True,
        help="Action to perform.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="If the metadata for the input shard exists, skip it.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of workers to use for processing.",
    )
    parser.add_argument(
        "--delete-cache-file",
        action="store_true",
        help="Delete the cache file if it exists.",
    )

    args = parser.parse_args()
    console = Console()

    protocol, _, path = args.dataset.partition("://")
    m_protocol, _, metadata_dir = args.metadata_dir.partition("://")
    if protocol != m_protocol:
        raise ValueError("Dataset and metadata store must be in the same filesystem.")

    fs, fs_path = fsspec.filesystem(protocol or "file"), path
    m_state_file = os.path.join(metadata_dir, "state.json")

    if fs.exists(m_state_file):
        try:
            raw_state = json.loads(fs.read_text(m_state_file))
        except json.JSONDecodeError:
            raise ValueError("Corrupted state file.")
        except FileNotFoundError:
            raw_state = {"shards": []}  # ???

        state = State(shards=set(raw_state["shards"]))
    else:
        state = State()

    fs_paths, skipped = [], 0
    for fs_path in braceexpand.braceexpand(fs_path):
        if fs_path in state.shards and not args.force:
            skipped += 1
            continue

        fs_paths.append(fs_path)

    if skipped:
        console.print(f"Skipping {skipped} cached shards.")

    console.print(f"Processing {len(fs_paths)} shards!")
    run_event_loop(
        console,
        args.action,
        state,
        fs,
        fs_paths,
        metadata_dir,
        args.num_workers,
    )


if __name__ == "__main__":
    main()