import multiprocessing as mp
from multiprocessing.synchronize import Event as Event_t
from dataclasses import dataclass as odataclass
from math import ceil
from os import remove, environ
from pathlib import Path
from shutil import rmtree
from yt_dlp import YoutubeDL
from yt_dlp.utils import download_range_func
from typing import Optional

from utils.constants import DEFAULT_CHUNKS_DIR, FFMPEG_PATH


@odataclass
class ChunkDownloadArgs:
    url:          str
    duration:     int
    chunk_size:   int
    start_offset: int


def get_out_path(index: int) -> Path:
    return DEFAULT_CHUNKS_DIR / f"chunk_{index}.mp4"


def download_video(url: str, out_path: Path, start_time: int, end_time: int):
    ffmpeg_path_s = str(FFMPEG_PATH.parent.absolute())
    environ["Path"] += f";{ffmpeg_path_s}"
    ydl_opts = {
        "format": "bestvideo[ext=mp4]/mp4",
        "quiet": True,
        "no_warnings": True,
        "outtmpl": str(out_path),
        "download_ranges": download_range_func(None, [(start_time, end_time)]),
        'nocheckcertificate': True
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def download_worker(dargs: ChunkDownloadArgs, next_event: Event_t, out_queue: mp.Queue) -> None:
    flip_index = 1

    duration = dargs.duration - dargs.start_offset
    nchunks = ceil(duration / dargs.chunk_size)
    for cidx in range(1, nchunks):
        next_event.wait()
        next_event.clear()

        out_path = get_out_path(flip_index)
        start = dargs.start_offset + cidx * dargs.chunk_size
        chunk_size = min(dargs.duration - start, dargs.chunk_size)

        download_video(dargs.url, out_path, start, start+chunk_size-1)
        out_queue.put(out_path)

        if (p := get_out_path((flip_index-2)%3)).exists():
            remove(p)

        flip_index = (flip_index + 1) % 3
        


class ChunkFetcher:
    def __init__(self, url: str, duration: int, chunk_size: int, chunk_offset: int = 0) -> None:
        self.dargs = ChunkDownloadArgs(
            url=url,
            duration=duration,
            chunk_size=chunk_size,
            start_offset=chunk_offset
        )
        self.cidx = 0
        self.num_chunks = ceil(duration / chunk_size)

        self.next_event = mp.Event()
        self.out_queue = mp.Queue(maxsize=3)

        self.download_process = mp.Process(target=download_worker,
                                      args=(self.dargs, self.next_event, self.out_queue))

        if DEFAULT_CHUNKS_DIR.exists():
            rmtree(DEFAULT_CHUNKS_DIR)
        else:
            DEFAULT_CHUNKS_DIR.mkdir()
        self.download_process.start()

    def prepare_first(self) -> None:
        out_path = get_out_path(0)
        print("Info: Downloading chunk 0 ...")
        download_video(self.dargs.url,
                       out_path,
                       self.dargs.start_offset,
                       self.dargs.start_offset+self.dargs.chunk_size)
        self.out_queue.put(out_path)

    def prepare_next(self) -> None:
        self.next_event.set()

    def get_next_path(self) -> Optional[Path]:
        if self.cidx > self.num_chunks:
            return None

        next_path = self.out_queue.get(block=True) # timeout=120.0)
        return next_path
    
    def stop(self) -> None:
        self.download_process.join()
