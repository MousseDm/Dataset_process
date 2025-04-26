#
# For licensing see accompanying LICENSE.txt file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--downloads_dir", required=True)
parser.add_argument("--decompress_dir")
parser.add_argument("--delete_archive_after_decompress", action="store_true")
args = parser.parse_args()



print("[HYPERSIM: DATASET_DOWNLOAD_IMAGES] Begin...")



if not os.path.exists(args.downloads_dir): os.makedirs(args.downloads_dir)

if args.decompress_dir is not None:
    if not os.path.exists(args.decompress_dir): os.makedirs(args.decompress_dir)



def download(url):
    download_name = os.path.basename(url)
    download_file = os.path.join(args.downloads_dir, download_name)

    cmd = "curl " + url + " --output " + download_file
    print("")
    print(cmd)
    print("")
    retval = os.system(cmd)
    assert retval == 0

    if args.decompress_dir is not None:
        cmd = "unzip " + download_file + " -d " + args.decompress_dir
        print("")
        print(cmd)
        print("")
        retval = os.system(cmd)
        assert retval == 0
        if args.delete_archive_after_decompress:
            cmd = "rm " + download_file
            print("")
            print(cmd)
            print("")
            retval = os.system(cmd)
            assert retval == 0



urls_to_download = [
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_001_001.zip",
]

for url in urls_to_download:
    download(url)

print("[HYPERSIM: DATASET_DOWNLOAD_IMAGES] Finished.")