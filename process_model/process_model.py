import sys
from typing import Dict, Optional, Tuple

import re
from jinja2 import PackageLoader, Environment
import click


replacer = re.compile(r"^DT")

env = Environment(
    loader=PackageLoader(__name__),
    lstrip_blocks=True,
    trim_blocks=True,
)


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
    help="Where to send Fortran output.",
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
    help="Indentation level of output code (default 4)",
)
def main(model_dirs, output_file, tag_set, module_name, signature_def, indent):
    """
    Utility to read a SavedModel TensorFlow model and export the necessary
    Fortran code to allow it to be interfaced with the fortran-tf library.

    Each MODEL_DIR contains a TensorFlow SavedModel.
    """

    from tensorflow.python.tools import saved_model_utils

    models = []
    for model_dir in model_dirs:
        # Tag-sets
        tag_sets = saved_model_utils.get_saved_model_tag_sets(model_dir)
        tags = tag_set.split(",")

        for ts in sorted(tag_sets):
            tag_string = ",".join(sorted(ts))
            if tag_string == tag_set:
                break
        else:
            print(
                f"The SavedModel {model_dir}\ndoes not contain tag-set " '"{tag_set}".',
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
                print('SignatureDef key: "%s"' % signature_def_key, file=sys.stderr)
            sys.exit(1)

        # Input and output tensors.
        input_tensors = signature_def_map[signature_def].inputs
        output_tensors = signature_def_map[signature_def].outputs

        # Get the info on these tensors: name, index, data type, and shape.
        inputs = _extract_tensor_info(input_tensors)
        outputs = _extract_tensor_info(output_tensors)

        models.append(Model(model_dir, tags, inputs, outputs))

    # Start generating Fortran.
    output_str = _render_template(
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

    for key, function_str in sorted(Model.functions_needed.items()):
        output_str += function_str

    # Expand the tabs and send the output to the file.
    output_str += "\nend module {}".format(module_name)
    output_str = output_str.replace("\t", " " * indent)

    print(output_str, file=output_file)


def _extract_tensor_info(tensors):
    """
    Recovers the useful info from the SavedModel tensor datatype.
    """
    from tensorflow.core.framework import types_pb2

    tf_types = {value: key for (key, value) in types_pb2.DataType.items()}

    tensor_info = {}
    for k, v in tensors.items():
        tensor_info[k] = {
            "name": v.name[: v.name.index(":")],
            "index": int(v.name[v.name.index(":") + 1 :]),
            "dtype": replacer.sub("TF", tf_types[v.dtype]),
            "tensor_shape": [],
        }
        for sz in v.tensor_shape.dim:
            tensor_info[k]["tensor_shape"].append(sz.size)

    return tensor_info


def _render_template(template_name, **kwargs):
    """
    Renders a template, replaces the tabs with `indent` number of spaces,
    returns the result.
    """
    t = env.get_template(template_name)
    s = t.render(kwargs)
    return s


def _map_tf_type_to_fortran(tf_type: str) -> Optional[Tuple[str, str]]:
    """
        Return a Fortran type that matches the Tensorflow type and a prefix
        for the Fortran function.
        E.g. TF_FLOAT -> real(kind=c_float), r32

        The r32 part will help form the Fortran associate_tensor function name:
        E.g. TF_FLOAT, [:,40] -> r32_2_associate_tensor


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
    datatypes: Dict[str : Optional[Tuple[str, str]]] = {
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

    functions_needed = {}
    longest_tag_len = 0

    def __init__(self, model_dir, tags, inputs, outputs):
        self.model_dir = model_dir
        self.tags = tags
        for tag in tags:
            if len(tag) > Model.longest_tag_len:
                Model.longest_tag_len = len(tag)
        self.inputs = inputs
        self.outputs = outputs
        for key, inp in sorted(inputs.items()):
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
                    "{}_{}_associate_tensor".format(tup[1], len(inp["tensor_shape"]))
                ] = assoc_def_str


if __name__ == "__main__":
    main()
