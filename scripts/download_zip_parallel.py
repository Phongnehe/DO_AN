"""Resume the requested public Kaggle ZIP using verified HTTP byte ranges."""
import concurrent.futures
import json
import os
import threading
import time
import urllib.request
from pathlib import Path

URL = 'https://www.kaggle.com/api/v1/datasets/download/nguynquy/macsaa?datasetVersionNumber=13'
DESTINATION = Path('/private/tmp/mabsa-vimacsa-v13.zip')
PROGRESS = Path('/private/tmp/mabsa-vimacsa-v13-download.json')


def main():
    request = urllib.request.Request(URL, headers={'Range': 'bytes=0-0'})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 206:
            raise RuntimeError('Byte ranges not supported')
        final_url = response.url
        total = int(response.headers['Content-Range'].rsplit('/', 1)[1])
        response.read()
    if PROGRESS.exists():
        state = json.loads(PROGRESS.read_text())
        if state['total'] != total:
            raise RuntimeError('Archive size changed; refusing to combine versions')
    else:
        state = {'total': total, 'prefix': DESTINATION.stat().st_size if DESTINATION.exists() else 0,
                 'completed': []}
        PROGRESS.write_text(json.dumps(state))
    chunk_size = 32 * 1024 * 1024
    ranges = [(i, start, min(start + chunk_size, total) - 1)
              for i, start in enumerate(range(state['prefix'], total, chunk_size))]
    completed = set(state['completed'])
    pending = [r for r in ranges if r[0] not in completed]
    lock = threading.Lock()
    fd = os.open(DESTINATION, os.O_RDWR | os.O_CREAT, 0o600)
    os.ftruncate(fd, total)
    started = time.time()

    def download(task):
        index, start, end = task
        for attempt in range(5):
            try:
                request = urllib.request.Request(final_url, headers={'Range': f'bytes={start}-{end}'})
                with urllib.request.urlopen(request, timeout=90) as response:
                    expected = f'bytes {start}-{end}/{total}'
                    if response.status != 206 or response.headers.get('Content-Range') != expected:
                        raise RuntimeError('Incorrect byte range response')
                    offset = start
                    while True:
                        data = response.read(1024 * 1024)
                        if not data:
                            break
                        if offset + len(data) > end + 1:
                            raise RuntimeError('Range response exceeds requested length')
                        written = 0
                        while written < len(data):
                            written += os.pwrite(fd, data[written:], offset + written)
                        offset += len(data)
                    if offset != end + 1:
                        raise RuntimeError('Incomplete byte range')
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        with lock:
            completed.add(index)
            state['completed'] = sorted(completed)
            temp = PROGRESS.with_suffix('.tmp')
            temp.write_text(json.dumps(state))
            os.replace(temp, PROGRESS)
            done = state['prefix'] + sum(e - s + 1 for i, s, e in ranges if i in completed)
            print(f'Download {done / 1024**2:.0f}/{total / 1024**2:.0f} MiB, {100 * done / total:.1f}%, {time.time() - started:.0f}s', flush=True)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(download, pending))
        os.fsync(fd)
    finally:
        os.close(fd)
    print('Local archive download complete.', flush=True)


if __name__ == '__main__':
    main()
