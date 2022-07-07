from pathlib import Path
import zipfile
import os
import pytest
from pytest_console_scripts import ScriptRunner


@pytest.fixture
def leaves_example_data_path(tmp_path):
    zip_data_path = Path(__file__).parent.parent.parent / \
        'docs/example_data/leaves_example_data.zip'
    assert zip_data_path.exists()
    with zipfile.ZipFile(zip_data_path, 'r') as zip_ref:
        zip_ref.extractall(tmp_path)
    return tmp_path


def test_leafexample_train(leaves_example_data_path, script_runner: ScriptRunner):
    # "'{}/*.tif'".format(str(leaves_example_data_path))
    image_path = str(leaves_example_data_path)
    label_path = "{}/leaf_labels_ilastik133.ilp".format(
        str(leaves_example_data_path))
    assert Path(label_path).exists()

    ret = script_runner.run('yapic', 'train', 'unet_2d',
                            image_path, label_path, '-e', '1', '--steps', '2')

    assert ret.success
