"""Minimal patches so the official test scripts run on a current Colab stack.

Only compatibility fixes, nothing that changes the predictions:
  PraNet    MyTest.py         scipy.misc.imsave (removed in SciPy 1.2) -> cv2.imwrite, as Polyp-PVT does
            MyTest.py         F.upsample (deprecated) -> F.interpolate, load weights on any device
            PraNet_Res2Net.py do not load ImageNet Res2Net from the authors' local path;
                              PraNet-19.pth already contains the backbone weights
  Polyp-PVT Test.py           drop unused scipy.misc import, F.upsample -> F.interpolate

Usage: python scripts/patch_repos.py <PraNet dir> <Polyp-PVT dir>
The script is idempotent and fails loudly if an expected line is missing.
"""

import sys
from pathlib import Path


def patch(path, replacements):
    text = path.read_text()
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)
        elif new not in text:
            sys.exit(f"{path}: expected to find {old!r}")
    path.write_text(text)
    print(f"patched {path}")


def main(pranet, pvt):
    pranet, pvt = Path(pranet), Path(pvt)

    patch(pranet / "MyTest.py", [
        ("from scipy import misc", "import cv2"),
        ("misc.imsave(save_path+name, res)", "cv2.imwrite(save_path+name, res*255)"),
        ("F.upsample(", "F.interpolate("),
        ("torch.load(opt.pth_path)", "torch.load(opt.pth_path, map_location='cpu')"),
    ])
    patch(pranet / "lib" / "PraNet_Res2Net.py", [
        ("res2net50_v1b_26w_4s(pretrained=True)", "res2net50_v1b_26w_4s(pretrained=False)"),
    ])

    patch(pvt / "Test.py", [
        ("from scipy import misc\n", ""),
        ("F.upsample(", "F.interpolate("),
        ("torch.load(opt.pth_path)", "torch.load(opt.pth_path, map_location='cpu')"),
    ])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
