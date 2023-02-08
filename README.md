# process_model
The process_model tool reads a TensorFlow SavedModel and outputs Fortran code to interface it to the fortran-tf-lib

## Installing
In a suitable Python environment do:
```
pip install git+https://github.com/Cambridge-ICCS/process_model.git
```
Note that as of 20/01/23 there is no tensorflow package in Pypi for Python >= 3.11.

## Running the tool
The pip install will place a `process_model` command in the PATH.  To use it, run it against one
or more TensorFlow SavedModel models.

```
process_model model_1 model_2 ...
```
The tool will output Fortran code to standard output, or to the file
specified with the `-o` option.

## Using the resulting Fortran
The output is a module, named `ml_module` by default.  It has procedures called
`ml_module_init`, `ml_module_calc`, `ml_module_finish`.
It also may have some `*_associate_tensor` routines tailored for the inputs
of the model.  So if the model expects a Tensor of type `TF_FLOAT` and of shape
`[-1, 40]` then there will be a `r32_2_associate_tensor` routine to generate
appropriately shaped and typed tensors from Fortran arrays.

The `ml_module_init` routine should be called once, before using `calc`.  It loads
the models into module variables.

## Worked example

## API reference
