import os
from click.testing import CliRunner
from process_model import main


def test_cli_works(tmp_path):
    """
    Test if process_model generates the Fortran file.
    """
    runner = CliRunner()
    result = runner.invoke(main)
    assert result.exit_code != 0
    result = runner.invoke(main, ['--help'])
    assert result.exit_code == 0
    assert 'Usage: main [OPTIONS] MODEL_DIR...' in result.output


def test_f90_created(tmp_path):
    """
    Test if process_model generates the Fortran file.
    """
    runner = CliRunner()
    output_path = tmp_path / 'testf.f90'
    result = runner.invoke(main, ['tests/my_model', '-o', output_path])
    assert result.exit_code == 0
    assert os.path.isfile(output_path)


def test_f90_tags_arg(tmp_path):
    """
    Test if process_model accepts tags argument correctly.
    """
    runner = CliRunner()
    output_path = tmp_path / 'testf.f90'
    result = runner.invoke(main, ['tests/my_model', '-t', 'unknown',
                                  '-o', output_path])
    assert not os.path.isfile(output_path)
    assert result.exit_code == 1
    result = runner.invoke(main, ['tests/my_model', '-t', 'serve',
                                  '-o', output_path])
    assert result.exit_code == 0
    assert os.path.isfile(output_path)


def test_f90_sig_def(tmp_path):
    """
    Test if process_model accepts signature_def argument correctly.
    """
    runner = CliRunner()
    output_path = tmp_path / 'testf.f90'
    result = runner.invoke(main, ['tests/my_model', '-s', 'unknown',
                                  '-o', output_path])
    assert not os.path.isfile(output_path)
    assert result.exit_code == 1
    result = runner.invoke(main, ['tests/my_model', '-s', 'serving_default',
                                  '-o', output_path])
    assert os.path.isfile(output_path)
    assert result.exit_code == 0
