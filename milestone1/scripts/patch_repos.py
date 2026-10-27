"""Small compatibility patches so the official test scripts run on a current PyTorch/SciPy.

None of them changes the predictions:
  PraNet/MyTest.py             scipy.misc.imsave no longer exists -> cv2.imwrite (as Polyp-PVT does);
                               F.upsample -> F.interpolate; load the weights on any device
  PraNet/lib/PraNet_Res2Net.py do not fetch ImageNet Res2Net weights from the authors' local path,
                               PraNet-19.pth already contains the backbone
  Polyp-PVT/Test.py            drop the unused scipy.misc import; F.upsample -> F.interpolate;
                               load the weights on any device

Usage: python scripts/patch_repos.py <PraNet dir> <Polyp-PVT dir>
Running it twice is fine. It stops with an error if a line it expects is not there.
"""

import sys
from pathlib import Path

PATCHES = {
    "MyTest.py": [
        ("from scipy import misc", "import cv2"),
        ("misc.imsave(save_path+name, res)", "cv2.imwrite(save_path+name, res*255)"),
        ("F.upsample(", "F.interpolate("),
        ("torch.load(opt.pth_path)", "torch.load(opt.pth_path, map_location='cpu')"),
    ],
    "lib/PraNet_Res2Net.py": [
        ("res2net50_v1b_26w_4s(pretrained=True)", "res2net50_v1b_26w_4s(pretrained=False)"),
    ],
    "Test.py": [
        ("from scipy import misc\n", ""),
        ("F.upsample(", "F.interpolate("),
        ("torch.load(opt.pth_path)", "torch.load(opt.pth_path, map_location='cpu')"),
    ],
}


def patch(path, replacements):
    text = path.read_text()
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)
        elif new not in text:  # neither the original nor the patched line: unexpected file
            sys.exit(f"{path}: expected to find {old!r}")
    path.write_text(text)
    print(f"patched {path}")


def main(pranet_dir, pvt_dir):
    pranet, pvt = Path(pranet_dir), Path(pvt_dir)
    patch(pranet / "MyTest.py", PATCHES["MyTest.py"])
    patch(pranet / "lib" / "PraNet_Res2Net.py", PATCHES["lib/PraNet_Res2Net.py"])
    patch(pvt / "Test.py", PATCHES["Test.py"])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
