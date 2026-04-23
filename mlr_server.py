from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from server.async_server import main


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--base-dir" not in argv:
        argv = ["--base-dir", str(SCRIPT_DIR)] + argv
    raise SystemExit(main(argv))
