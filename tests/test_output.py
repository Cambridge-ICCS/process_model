import os
import re
from click.testing import CliRunner
from process_model import main


def test_assoc_tensors(tmp_path):
    """
    Test that code generated for my_model has a function
    for associating real(:,:) arrays into a TF_Tensor.
    """
    runner = CliRunner()
    output_path = tmp_path / 'testf.f90'
    result = runner.invoke(main, ['tests/my_model'])
    pattern_funcdef = re.compile(
        r'^    function r32_2_associate_tensor\(input_array, input_shape,'
        r' input_size\)',
        re.MULTILINE
    )
    assert pattern_funcdef.search(result.output)
    pattern_inptype = re.compile(
        r'^        real\(kind=c_float\), dimension\(:, :\), target'
        r' :: input_array',
        re.MULTILINE
    )
    assert pattern_inptype.search(result.output)
