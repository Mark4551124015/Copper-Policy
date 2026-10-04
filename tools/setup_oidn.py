"""Install the official OIDN 2.3.3 Blackwell fix inside the project uv .venv."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = ROOT / ".venv"
VERSION = "2.3.3"
URL = f"https://github.com/RenderKit/oidn/releases/download/v{VERSION}/oidn-{VERSION}.x86_64.linux.tar.gz"
# These official release libraries match the previously working environment.
HASHES = {
    "libOpenImageDenoise.so.2.3.3": "9ac9dcd18318cae879efb01c14cff72c2c4275428a15e5ab6e94c8ff8a43c4dc",
    "libOpenImageDenoise_core.so.2.3.3": "eadb2437d6ac4a88e5746980495ba1091f7d14745e3000b82a9382093d22f351",
    "libOpenImageDenoise_device_cuda.so.2.3.3": "12b3ff87cfa4b6a90481b757f5a56ee69380a45e3dcaeadf8dd22cb73d4f714a",
}

def main():
    if Path(sys.prefix).resolve() != ENVIRONMENT.resolve():
        raise SystemExit("Run with the project's uv environment")
    spec = importlib.util.find_spec("sapien")
    if spec is None:
        raise SystemExit("Install the robotwin extra first")
    package = Path(next(iter(spec.submodule_search_locations)))
    if not package.resolve().is_relative_to(ENVIRONMENT.resolve()):
        raise SystemExit("SAPIEN must be inside the project .venv")
    target = package / "oidn_library"
    ready = all((target / name).is_file() and
                hashlib.sha256((target / name).read_bytes()).hexdigest() == checksum
                for name, checksum in HASHES.items())
    if not ready:
        cache = ROOT / ".uv-cache" / "oidn"
        cache.mkdir(parents=True, exist_ok=True)
        archive = cache / URL.rsplit("/", 1)[1]
        if not archive.exists():
            print(f"Downloading official OIDN: {URL}", flush=True)
            partial = archive.with_suffix(".partial")
            urllib.request.urlretrieve(URL, partial)
            partial.replace(archive)
        with tarfile.open(archive, "r:gz") as bundle:
            payloads = {}
            for name, checksum in HASHES.items():
                members = [m for m in bundle.getmembers() if m.isfile() and Path(m.name).name == name]
                if len(members) != 1:
                    raise SystemExit(f"Expected one official library: {name}")
                data = bundle.extractfile(members[0]).read()
                if hashlib.sha256(data).hexdigest() != checksum:
                    raise SystemExit(f"OIDN checksum mismatch: {name}; remove cached archive and retry")
                payloads[name] = data
            for name, data in payloads.items():
                temporary = target / (name + ".partial")
                temporary.write_bytes(data)
                temporary.replace(target / name)
    loader = package / "_oidn_tricks.py"
    text = loader.read_text()
    updated = text.replace(".so.2.0.1", ".so.2.3.3")
    if ".so.2.3.3" not in updated:
        raise SystemExit(f"Unexpected SAPIEN OIDN loader: {loader}")
    if text != updated:
        temporary = loader.with_name(loader.name + ".copper-patch")
        temporary.write_text(updated)
        temporary.replace(loader)
    print("OIDN 2.3.3 installed and SHA-256 verified; restart evaluation processes.")

if __name__ == "__main__":
    main()
