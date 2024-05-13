import os
import json
import argparse
import fsspec
import subprocess
import braceexpand
import tempfile
import concurrent.futures as cf
import multiprocessing as mp
from rich.progress import Progress
from rich.console import Console
from dataclasses import dataclass, field


WDS_TMP_DIR = os.path.join(tempfile.gettempdir(), "wds-data")
os.makedirs(WDS_TMP_DIR, exist_ok=True)

@dataclass
class State:
    shards: set[str] = field(default_factory=set)

    def to_json(self) -> str:
        data = {
            "shards": list(self.shards),
        }
        return json.dumps(data)


@dataclass
class StramingS3File:
    url: str
    tmp_file: str | None = None
    stream: object | None = None

    def __enter__(self):
        fd, self.tmp_file = tempfile.mkstemp(dir=WDS_TMP_DIR)
        subprocess.check_call(
            ["s5cmd", "cat", self.url],
            stdout=fd,
        )
        os.close(fd)
        self.stream = open(self.tmp_file, "rb")
        return self.stream

    def __exit__(self, exc_type, exc_value, traceback):
        assert self.tmp_file is not None
        os.remove(self.tmp_file)
        self.stream.close() # type: ignore



def wds_init():
    import webdataset.gopen as wds_gopen
    def gopen_s3_python(url, mode="rb", bufsize=8192):
        if not url.startswith("s3://"):
            raise ValueError("gopen_s3_python only works with s3:// urls")

        assert mode == "rb"
        return StramingS3File(url)

    wds_gopen.gopen_schemes["s3"] = gopen_s3_python


def extract_alt_fn(
    fs: fsspec.AbstractFileSystem,
    wds_file: str,
    metadata_file: str,
) -> None:
    import webdataset as wds

    wds_init()
    dataset = wds.WebDataset(fs.unstrip_protocol(wds_file)).decode().to_tuple("json")

    alt_texts = []
    for (metadata,) in dataset:
        alt_texts.append(metadata.get("caption"))

    print(f"Found {len(alt_texts)} samples in {wds_file}.")
    fs.write_text(metadata_file, json.dumps(alt_texts))


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
        executor = cf.ProcessPoolExecutor(  # type: ignore
            max_workers=num_workers,
            mp_context=mp.get_context("spawn"),
        )

    if action == "extract_alt":
        action_fn, target_ext = extract_alt_fn, ".json"
    elif action == "caption":
        action_fn, target_ext = caption_fn, ".json"
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
                fs_path_name, _ = os.path.splitext(fs_path)
                metadata_file = os.path.join(
                    metadata_dir,
                    f"{os.path.basename(fs_path_name)}{target_ext}",
                )

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
                    console.print(f"Error processing {fs_path}: {e}", style="red")
                    console.print_exception()
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
        choices=["extract_alt", "caption"],
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
