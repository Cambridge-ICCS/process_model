"""Code to output a Fortran module to interface to the given TF models.

process_model uses some of the methods provided in the TensorFlow
saved_model_cli app to inspect a TensorFlow SavedModel and output the necessary
Fortran to interface it to the fortran-tf-lib library.

"""

import sys
import re
import click
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple
from jinja2 import Environment, PackageLoader


replacer = re.compile(r"^DT")

env = Environment(
    loader=PackageLoader(__name__),
    lstrip_blocks=True,
    trim_blocks=True,
)

# TODO: Consider adding type hints to the function defs.
# TODO: Consider adding argument types and descriptions to docstrings.
# TODO: consider linting with mypy, pydoctyle (numpy convention?) and pylint.


@click.command()
@click.argument(
    "model_dirs",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    nargs=-1,
    required=True,
)
@click.option(
    "-o",
    "--output-file",
    default="-",
    type=click.File(mode="wt"),
    help="Where to save the Fortran output (default stdout).",
)
@click.option(
    "-t",
    "--tag-set",
    default="serve",
    type=click.STRING,
    help="tag-set to use (default `serve`).",
)
@click.option(
    "-m",
    "--module-name",
    default="ml_module",
    type=click.STRING,
    help="Name to give the Fortran module (default `ml_module`).",
)
@click.option(
    "-s",
    "--signature-def",
    default="serving_default",
    type=click.STRING,
    help="signature def to use (default " "`serving_default`)",
)
@click.option(
    "--indent",
    default=4,
    type=click.INT,
    help="Indentation level of output Fortran code (default 4)",
)
def main(
    model_dirs: Sequence[click.Path],
    output_file: TextIO,
    tag_set: str,
    module_name: str,
    signature_def: str,
    indent: int,
):
    """Main method for process_model.

    Parameters
    ----------
    model_dirs : List[click.Path]
        A list of click.Paths to directories containing TensorFlow SavedModels.
    output_file : click.File
        Where to save the generated Fortran.
    tag_set : click.STRING
        TensorFlow tag_set, e.g. 'serve,gpu', or 'serve' (the default).
    module_name : click.STRING
        The name to give the Fortran module (default 'ml_module').
    signature_def : click.STRING
        An opaque parameter for the TensorFlow model.
        Default 'serving_default'.
    indent : click.INT
        Indentation width in spaces for generated Fortran code (default 4).


    Utility to read a SavedModel TensorFlow model and export the necessary
    Fortran code to allow it to be interfaced with the fortran-tf library.

    Each MODEL_DIR contains a TensorFlow SavedModel.
    """

    # Put these imports here because tensorflow takes a _very_ long time to
    # load, and the user may just be doing a --help invocation.
    #
    # Look, don't blame me, blame TensorFlow.
    from tensorflow.python.tools import saved_model_utils

    models: List[Model] = []
    for model_dir in model_dirs:
        # Tag-sets
        tag_sets: Sequence[str] = saved_model_utils.get_saved_model_tag_sets(
            model_dir
        )
        tags = tag_set.split(",")

        for ts in sorted(tag_sets):
            tag_string = ",".join(sorted(ts))
            if tag_string == tag_set:
                break
        else:
            # Reach here when the for loop finishes normally.
            print(
                f"The SavedModel {model_dir}\ndoes not contain tag-set "
                '"{tag_set}".',
                file=sys.stderr,
            )
            print("It contains the following tag-sets:", file=sys.stderr)
            for ts in sorted(tag_sets):
                print("%r" % ",".join(sorted(ts)), file=sys.stderr)
            sys.exit(1)

        # Signature defs.
        meta_graph = saved_model_utils.get_meta_graph_def(model_dir, tag_set)
        signature_def_map = meta_graph.signature_def
        if signature_def not in signature_def_map:
            print(
                f"The SavedModel {model_dir}\ndoes not contain"
                ' signature-def "{signature_def}".',
                file=sys.stderr,
            )
            print(
                "The model's MetaGraphDef contains SignatureDefs with the "
                "following keys:",
                file=sys.stderr,
            )
            for signature_def_key in sorted(signature_def_map.keys()):
                print(
                    f'SignatureDef key: "{signature_def_key}"',
                    file=sys.stderr,
                )
            sys.exit(1)

        # Input and output tensors.
        input_tensors = signature_def_map[signature_def].inputs
        output_tensors = signature_def_map[signature_def].outputs

        # Get the info on these tensors: name, index, data type, and shape.
        inputs = _extract_tensor_info(input_tensors)
        outputs = _extract_tensor_info(output_tensors)

        models.append(Model(model_dir, tags, inputs, outputs))

    # Start generating Fortran.
    output_str: str = _render_template(
        "module_start.F90",
        functions_needed=Model.functions_needed,
        module_name=module_name,
        model_dirs=model_dirs,
    )

    # Keep the next part quite simple so it's easy to extend.

    # Pass through all the models we've scanned.
    for loop_index, model in enumerate(models, start=1):

        # Model definitions.
        output_str += _render_template(
            "module_definitions.F90",
            model_index=loop_index,
            model_dir=model.model_dir,
            inputs=model.inputs,
            outputs=model.outputs,
        )

    output_str += "\n\tcontains\n\n"

    output_str += _render_template(
        "module_body.F90",
        max_tags_len=Model.longest_tag_len,
        models=models,
        module_name=module_name,
    )

    for _, function_str in sorted(Model.functions_needed.items()):
        output_str += function_str

    # Expand the tabs and send the output to the file.
    output_str += f"\nend module {module_name}"
    output_str = output_str.replace("\t", " " * indent)

    print(output_str, file=output_file)


def _extract_tensor_info(tensors) -> Dict[str, Dict[str, Any]]:
    """
    Recovers the useful info from the SavedModel tensor datatype.
    """
    # Put these imports here because tensorflow takes a _very_ long time to
    # load, and the user may just be doing a --help invocation.
    #
    # Look, don't blame me, blame TensorFlow.
    #
    # Also the type stubs for tensorflow don't believe this package exists.
    from tensorflow.core.framework import types_pb2  # type: ignore

    tf_types = {value: key for (key, value) in types_pb2.DataType.items()}

    tensor_info: Dict[str, Dict[str, Any]] = {}
    for k, v in tensors.items():
        tensor_info[k] = {
            "name": v.name[: v.name.index(":")],
            "index": int(v.name[v.name.index(":") + 1:]),
            "dtype": replacer.sub("TF", tf_types[v.dtype]),
            "tensor_shape": [],
        }
        for sz in v.tensor_shape.dim:
            tensor_info[k]["tensor_shape"].append(sz.size)

    return tensor_info


def _render_template(template_name, **kwargs) -> str:
    """
    Renders a Jinja template.
    """
    template = env.get_template(template_name)
    output_string = template.render(kwargs)
    return output_string


def _map_tf_type_to_fortran(tf_type: str) -> Optional[Tuple[str, str]]:
    """
        Return a Fortran type that matches the Tensorflow type and a prefix
        for the Fortran function.
        E.g. TF_FLOAT -> real(kind=c_float), r32

        The r32 part will help form the Fortran associate_tensor function name:
        E.g. TF_FLOAT, [:,40] -> r32_2_associate_tensor

        Returns None if we don't know how to associate that kind of Tensor
        from Fortran.

        Data taken from include/tensorflow/c/tf_datatype.h:
    typedef enum TF_DataType {
      TF_FLOAT = 1,
      TF_DOUBLE = 2,
      TF_INT32 = 3,  // Int32 tensors are always in 'host' memory.
      TF_UINT8 = 4,
      TF_INT16 = 5,
      TF_INT8 = 6,
      TF_STRING = 7,
      TF_COMPLEX64 = 8,  // Single-precision complex
      TF_COMPLEX = 8,    // Old identifier kept for API backwards compatibility
      TF_INT64 = 9,
      TF_BOOL = 10,
      TF_QINT8 = 11,     // Quantized int8
      TF_QUINT8 = 12,    // Quantized uint8
      TF_QINT32 = 13,    // Quantized int32
      TF_BFLOAT16 = 14,  // Float32 truncated to 16 bits.  Only for cast ops.
      TF_QINT16 = 15,    // Quantized int16
      TF_QUINT16 = 16,   // Quantized uint16
      TF_UINT16 = 17,
      TF_COMPLEX128 = 18,  // Double-precision complex
      TF_HALF = 19,
      TF_RESOURCE = 20,
      TF_VARIANT = 21,
      TF_UINT32 = 22,
      TF_UINT64 = 23,
    } TF_DataType;
    """
    datatypes: Dict[str, Optional[Tuple[str, str]]] = {
        "TF_FLOAT": ("real(kind=c_float)", "r32"),
        "TF_DOUBLE": ("real(kind=c_double)", "r64"),
        "TF_INT32": ("integer(kind=c_int32_t)", "i32"),
        "TF_UINT8": ("integer(kind=c_int8_t)", "i8"),
        "TF_INT16": ("integer(kind=c_int16_t)", "i16"),
        "TF_INT8": ("integer(kind=c_int8_t)", "i8"),
        "TF_STRING": ("character(:)", "ch"),
        "TF_COMPLEX64": ("complex(kind=c_float_complex)", "c64"),
        "TF_COMPLEX": ("complex(kind=c_float_complex)", "c64"),
        "TF_INT64": ("integer(kind=c_int64_t)", "i64"),
        "TF_BOOL": ("logical(kind=c_bool)", "l"),
        "TF_QINT8": None,
        "TF_QUINT8": None,
        "TF_QINT32": None,
        "TF_BFLOAT16": None,
        "TF_QINT16": None,
        "TF_QUINT16": None,
        "TF_UINT16": ("integer(kind=c_int16_t)", "i16"),
        "TF_COMPLEX128": ("complex(kind=c_double_complex)", "c128"),
        "TF_HALF": None,
        "TF_RESOURCE": None,
        "TF_VARIANT": None,
        "TF_UINT32": ("integer(kind=c_int32_t)", "i32"),
        "TF_UINT64": ("integer(kind=c_int64_t)", "i64"),
    }
    return datatypes[tf_type]


class Model:
    """
    Utility class to hold info about each ML model.
    """

    # Class attributes.
    functions_needed: Dict[str, str] = {}
    longest_tag_len = 0

    def __init__(self, model_dir, tags, inputs, outputs):
        self.model_dir = model_dir
        self.tags = tags
        for tag in tags:
            if len(tag) > Model.longest_tag_len:
                Model.longest_tag_len = len(tag)
        self.inputs = inputs
        self.outputs = outputs
        for _, inp in sorted(inputs.items()):
            tup = _map_tf_type_to_fortran(inp["dtype"])
            if tup:
                input_shape_len = len(inp["tensor_shape"])
                input_dimension_str = ", ".join(":" * input_shape_len)
                assoc_def_str = _render_template(
                    "associate_tensors.F90",
                    prefix=tup[1],
                    input_shape_len=input_shape_len,
                    input_type_dec=tup[0],
                    input_dimension_str=input_dimension_str,
                )
                Model.functions_needed[
                    f"{tup[1]}_{len(inp['tensor_shape'])}_associate_tensor"
                ] = assoc_def_str


if __name__ == "__main__":
    main()
