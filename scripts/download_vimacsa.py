"""Download the requested Kaggle dataset without storing a multi-GB archive.

Annotations are kept unchanged. Images are resized to the model's 224x224
input size and saved as JPEG; the manifest preserves original dimensions
so that RoI boxes can still be normalized correctly.
"""
import argparse
import io
import json
import time
import urllib.request
import zipfile
from collections import OrderedDict
from pathlib import Path

from PIL import Image


class RemoteArchive(io.RawIOBase):
    def __init__(self, url, block_size=8 * 1024 * 1024):
        request = urllib.request.Request(url, headers={'Range': 'bytes=0-0'})
        with urllib.request.urlopen(request, timeout=60) as response:
            self.url = response.url
            content_range = response.headers.get('Content-Range', '')
            if response.status != 206 or '/' not in content_range:
                raise RuntimeError('Server does not support range downloads; no archive was saved.')
            self.length = int(content_range.rsplit('/', 1)[1])
            response.read()
        self.position = 0
        self.block_size = block_size
        self.cache = OrderedDict()
        self.bytes_downloaded = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence == 0:
            self.position = offset
        elif whence == 1:
            self.position += offset
        elif whence == 2:
            self.position = self.length + offset
        return self.position

    def _block(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        start = index * self.block_size
        end = min(start + self.block_size, self.length) - 1
        for attempt in range(5):
            try:
                request = urllib.request.Request(self.url, headers={'Range': f'bytes={start}-{end}'})
                with urllib.request.urlopen(request, timeout=90) as response:
                    if response.status != 206:
                        raise RuntimeError(f'Expected range response, got {response.status}')
                    data = response.read()
                if len(data) != end - start + 1:
                    raise RuntimeError('Incomplete range response')
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        self.bytes_downloaded += len(data)
        self.cache[index] = data
        if len(self.cache) > 3:
            self.cache.popitem(last=False)
        return data

    def read(self, size=-1):
        size = self.length - self.position if size < 0 else min(size, self.length - self.position)
        chunks = []
        while size > 0:
            index, offset = divmod(self.position, self.block_size)
            block = self._block(index)
            chunk = block[offset:offset + size]
            if not chunk:
                break
            chunks.append(chunk)
            self.position += len(chunk)
            size -= len(chunk)
        return b''.join(chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/vimacsa')
    ap.add_argument('--max-images', type=int, default=7)
    ap.add_argument('--archive', default=None, help='existing local Kaggle zip')
    ap.add_argument('--image-dir', default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = 'https://www.kaggle.com/api/v1/datasets/download/nguynquy/macsaa?datasetVersionNumber=13'
    remote = RemoteArchive(source) if not args.archive else None
    archive_length = remote.length if remote else Path(args.archive).stat().st_size
    print(f'Archive size: {archive_length / 1024**3:.2f} GiB.', flush=True)
    with zipfile.ZipFile(remote or args.archive) as archive:
        files = {Path(info.filename).name: info for info in archive.infolist() if not info.is_dir()}
        wanted = set()
        for name in ['train.json', 'dev.json', 'test.json', 'roi_data.csv']:
            info = files[name]
            data = archive.read(info)
            (out / name).write_bytes(data)
            print(f'Saved annotations: {name} ({len(data)} bytes)', flush=True)
            if name.endswith('.json'):
                rows = json.loads(data)
                for row in rows:
                    wanted.update(row['list_img'][:args.max_images])
        manifest_path = out / 'image_manifest.json'
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        image_dir = Path(args.image_dir) if args.image_dir else out / 'image224'
        image_dir.mkdir(exist_ok=True)
        images = sorted([files[name] for name in wanted], key=lambda info: info.header_offset)
        print(f'Preparing {len(images)} images at 224x224; annotations and official splits preserved.', flush=True)
        for index, info in enumerate(images, 1):
            name = Path(info.filename).name
            destination = image_dir / (Path(name).stem + '.jpg')
            if name not in manifest or not destination.exists():
                with Image.open(io.BytesIO(archive.read(info))) as image:
                    width, height = image.size
                    image.convert('RGB').resize((224, 224), Image.Resampling.BICUBIC).save(destination, quality=90)
                manifest[name] = {'path': str(destination.resolve()), 'width': width, 'height': height}
            if index % 100 == 0 or index == len(images):
                manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding='utf-8')
                network_bytes = remote.bytes_downloaded if remote else archive_length
                print(f'Images {index}/{len(images)}; archive read/downloaded {network_bytes / 1024**2:.0f} MiB', flush=True)
    (out / 'download_info.json').write_text(json.dumps({
        'source': 'https://www.kaggle.com/datasets/nguynquy/macsaa',
        'version': 13, 'archive_bytes': archive_length,
        'downloaded_bytes': remote.bytes_downloaded if remote else archive_length,
        'max_images_per_review': args.max_images,
        'images': len(manifest), 'image_processing': 'RGB, BICUBIC resize 224x224, JPEG quality 90',
        'annotation_splits': 'Official train/dev/test, unchanged',
    }, indent=2), encoding='utf-8')
    print('Dataset download complete.', flush=True)


if __name__ == '__main__':
    main()
