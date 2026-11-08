"""Create a self-contained Arduino sketch folder using the current shared helpers."""
import argparse
from pathlib import Path
import shutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True,
                    help='New folder named handshake_only')
args = parser.parse_args()
if args.output.name != 'handshake_only':
    parser.error('output folder must be named handshake_only')
args.output.mkdir(parents=True, exist_ok=False)
source = Path(__file__).resolve().parent
for name in ('armlink.c', 'armlink.h'):
    shutil.copy2(source / name, args.output / name)
shutil.copy2(source / 'handshake_only' / 'handshake_only.ino', args.output)
print('Open in Arduino IDE:', args.output / 'handshake_only.ino')
