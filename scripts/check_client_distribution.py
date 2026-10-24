"""Check the independently distributed client's license and test data."""

import argparse
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path


def check(directory):
    wheels = list(directory.glob("wallx_rebot_client-*.whl"))
    sources = list(directory.glob("wallx_rebot_client-*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise ValueError("Expected exactly one client wheel and one source archive")
    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()
        metadata = BytesParser().parsebytes(
            archive.read(next(n for n in names if n.endswith(".dist-info/METADATA")))
        )
        if metadata["License-Expression"] != "Apache-2.0":
            raise ValueError("Client wheel must declare Apache-2.0")
        for name in ("LICENSE", "NOTICE.txt"):
            entry = next((n for n in names if n.endswith("/licenses/" + name)), None)
            if entry is None or not archive.read(entry).strip():
                raise ValueError(f"Client wheel is missing {name}")
    with tarfile.open(sources[0]) as archive:
        names = {n.split("/", 1)[-1] for n in archive.getnames()}
        for name in (
            "LICENSE",
            "NOTICE.txt",
            "docs/examples/server_manifest.json",
            "tests/fixtures/rtc_5c8ced1/steady-fifo.json",
        ):
            if name not in names:
                raise ValueError(f"Client source archive is missing {name}")
    print("Client wheel license and source archive fixtures: valid")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    check(parser.parse_args().directory)
