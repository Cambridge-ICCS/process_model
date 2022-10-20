import subprocess
import os
import click


def test_f90_created(tmp_path):
    """
    Test if process_model generates the Fortran file.
    """
    subprocess.run(["process_model", "tests/my_model/", "-o",
                    tmp_path / "testf.f90"], check=True)
    assert os.path.isfile(tmp_path / "testf.f90")


def test_f90_tags_arg(tmp_path):
    """
    Test if process_model accepts tags argument correctly.
    """
    ret = subprocess.run(["process_model", "-t", "unknown", "tests/my_model/",
                          "-o", tmp_path / "testf.f90"])
    assert not os.path.isfile(tmp_path / "testf.f90")
    assert ret.returncode == 1
    subprocess.run(["process_model", "-t", "serve", "tests/my_model/",
                    "-o", tmp_path / "testf.f90"], check=True)
    assert os.path.isfile(tmp_path / "testf.f90")


def test_f90_sig_def(tmp_path):
    """
    Test if process_model accepts signature_def argument correctly.
    """
    ret = subprocess.run(["process_model", "-s", "unknown", "tests/my_model/",
                          "-o", tmp_path / "testf.f90"])
    assert ret.returncode == 1
    assert not os.path.isfile(tmp_path / "testf.f90")
    subprocess.run(["process_model", "-s", "serving_default",
                    "tests/my_model/", "-o", tmp_path / "testf.f90"],
                   check=True)
    assert os.path.isfile(tmp_path / "testf.f90")
