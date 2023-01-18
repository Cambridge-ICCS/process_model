module {{ module_name }}
	use TF_Types
	use TF_Interface

	implicit none

	public {{ module_name }}_init, &
		{{ module_name }}_calc, &
		{{ module_name }}_finish, &
		associate_tensor

    {% if functions_needed %}
	! Interface for `associate_tensor` functions
	interface associate_tensor
    {% for func_name in functions_needed.keys() %}
		module procedure {{ func_name }}
    {% endfor %}
	end interface associate_tensor
    {% endif %}

	! Each model needs a session and a graph variable.

