import unittest
import tempfile
import os
import numpy as np

from bfiocpp import TSWriter, TSReader, Seq, FileType


import requests, pathlib, shutil, logging, sys
from ome_zarr.utils import download as zarr_download

TEST_IMAGES = {
    "5025551.zarr": "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0054A/5025551.zarr",
    "p01_x01_y01_wx0_wy0_c1.ome.tif": "https://raw.githubusercontent.com/sameeul/polus-test-data/main/bfio/p01_x01_y01_wx0_wy0_c1.ome.tif",
    "Plate1-Blue-A-12-Scene-3-P3-F2-03.czi": "https://downloads.openmicroscopy.org/images/Zeiss-CZI/idr0011/Plate1-Blue-A_TS-Stinger/Plate1-Blue-A-12-Scene-3-P3-F2-03.czi",
}

TEST_DIR = pathlib.Path(__file__).with_name("data")

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("bfio.test")

if "-v" in sys.argv:
    logger.setLevel(logging.INFO)


def setUpModule():
    """Download images for testing"""
    TEST_DIR.mkdir(exist_ok=True)

    for file, url in TEST_IMAGES.items():
        logger.info(f"setup - Downloading: {file}")

        if not file.endswith(".zarr"):
            if TEST_DIR.joinpath(file).exists():
                continue

            r = requests.get(url)

            with open(TEST_DIR.joinpath(file), "wb") as fw:
                fw.write(r.content)
        else:
            if TEST_DIR.joinpath(file).exists():
                shutil.rmtree(TEST_DIR.joinpath(file))
            zarr_download(url, str(TEST_DIR))

    """Load the czi image, and save as a npy file for further testing."""
    with bfio.BioReader(
        TEST_DIR.joinpath("Plate1-Blue-A-12-Scene-3-P3-F2-03.czi")
    ) as br:
        with bfio.BioWriter(
            TEST_DIR.joinpath("4d_array.ome.tif"),
            metadata=br.metadata,
            X=br.X,
            Y=br.Y,
            Z=br.Z,
            C=br.C,
            T=br.T,
        ) as bw:
            bw[:] = br[:]

class TestZarrWrite(unittest.TestCase):

    def test_write_zarr_2d(self):
        """test_write_zarr_2d - Write zarr using TSWrtier"""

        br = TSReader(
            str(TEST_DIR.joinpath("5025551.zarr/0")),
            FileType.OmeZarr,
            "",
        )
        assert br._X == 2702
        assert br._Y == 2700
        assert br._Z == 1
        assert br._C == 27
        assert br._T == 1

        rows = Seq(0, br._Y - 1, 1)
        cols = Seq(0, br._X - 1, 1)
        layers = Seq(0, 0, 1)
        channels = Seq(0, 0, 1)
        tsteps = Seq(0, 0, 1)
        tmp = br.data(rows, cols, layers, channels, tsteps)

        with tempfile.TemporaryDirectory() as test_dir:
            # Use the temporary directory
            test_file_path = os.path.join(test_dir, 'out/test.ome.zarr')

            bw = TSWriter(test_file_path)
            bw.write_image(tmp, tmp.shape, tmp.shape)
            bw.close()

            br = TSReader(
                str(test_file_path),
                FileType.OmeZarr,
                "",
            )

            rows = Seq(0, br._Y - 1, 1)
            cols = Seq(0, br._X - 1, 1)
            layers = Seq(0, 0, 1)
            channels = Seq(0, 0, 1)
            tsteps = Seq(0, 0, 1)
            tmp = br.data(rows, cols, layers, channels, tsteps)

            assert tmp.dtype == np.uint8
            assert tmp.sum() == 183750394
            assert tmp.shape == (1, 1, 1, 2700, 2702)

