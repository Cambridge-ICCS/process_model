	! Model: {{ model_dir }}
	type(TF_Session) :: model_session_{{ model_index }}
	type(TF_Graph) :: model_graph_{{ model_index }}

	! Input and output details
	type(TF_Output), dimension({{ inputs | length }}) :: inputs_{{ model_index }}
	type(TF_Output), dimension({{ outputs | length }}) :: outputs_{{ model_index }}

