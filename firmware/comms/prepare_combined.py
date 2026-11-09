"""Package the checkpoint 3A sketch and its shared helpers for Arduino IDE."""
import argparse
from pathlib import Path
import shutil

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
if a.output.name != 'combined_only':
    p.error('output folder must be named combined_only')
a.output.mkdir(parents=True, exist_ok=False)
root = Path(__file__).resolve().parent
for source in (root/'armlink.c', root/'armlink.h',
               root/'combined_only'/'combined_only.ino',
               root/'combined_only'/'servo_bridge.h'):
    shutil.copy2(source, a.output / source.name)
print('Open in Arduino IDE:', a.output/'combined_only.ino')
