	subroutine {{ module_name }}_init()
		! Filenames for directories containing models
		character({{ models | map(attribute='model_dir') | map('length') | max }}), dimension({{ models | length }}) :: model_dirs

		character({{ max_tags_len }}), dimension({{ models | map(attribute='tags') | map('length') | max }}, {{ models | length }}) :: tags
		integer :: i

		! Assign the tags
        {% for model in models %}
        {% set outer_loop = loop %}
        {% for tag in model['tags'] %}
		tags({{ loop.index }}, {{ outer_loop.index }}) = '{{ tag }}'
        {% endfor %}
        {% endfor %}

		! Rather than hard-coding the filenames here, you probably
		! want to load them from a config file or similar.
        {% for model in models %}
		model_dirs({{ loop.index }}) = '{{ model.model_dir }}'
        {% endfor %}

		! Load all the models.
		! If you have a model with different needs (tags, etc)
		! edit this to handle that model separately.

        {% for model in models %}
		! Model: {{ model['model_dir'] }}
		model_graph_{{ loop.index }} = TF_NewGraph()
		call load_model(model_session_{{ loop.index }}, &
			model_graph_{{ loop.index }}, &
			tags(:, {{ loop.index }}), model_dirs({{ loop.index }}))
        {% endfor %}

		! Populate the input / output operations.
        {% for model in models %}
        {% set outer_loop = loop %}
        {% for name, input in model.inputs | dictsort %}
		! Input for '{{ model.model_dir }}' input '{{ name }}'
		inputs_{{ outer_loop.index }}({{ loop.index }})%oper = TF_GraphOperationByName( &
			model_graph_{{ outer_loop.index }}, &
			'{{ input['name'] }}' &
		)
		if (.not.c_associated(inputs_{{ outer_loop.index }}({{ loop.index }})%oper%p)) then
			write(*,*)'inputs_{{ outer_loop.index }}({{ loop.index }}) not associated'
			stop
		endif
		inputs_{{ outer_loop.index }}({{ loop.index }})%index = {{ input['index'] }}

        {% endfor %}
        {% for name, output in model.outputs | dictsort %}
		! Output for '{{ model.model_dir }}' output '{{ name }}'
		outputs_{{ outer_loop.index }}({{ loop.index }})%oper = TF_GraphOperationByName( &
    		model_graph_{{ outer_loop.index }}, &
			'{{ output['name'] }}' &
		)
		if (.not.c_associated(outputs_{{ outer_loop.index }}({{ loop.index }})%oper%p)) then
			write(*,*)'outputs_{{ outer_loop.index }}({{ loop.index }}) not associated'
			stop
		endif
		outputs_{{ outer_loop.index }}({{ loop.index }})%index = {{ output['index'] }}

        {% endfor %}
        {% endfor %}
	end subroutine {{ module_name }}_init

	subroutine load_model(session, graph, tags, model_dir)

		type(TF_Session) :: session
		type(TF_Graph) :: graph
		character(*), dimension(:) :: tags
		character(*) :: model_dir

		type(TF_SessionOptions) :: sessionoptions
		type(TF_Status) :: stat
		character(100) :: message

		sessionoptions = TF_NewSessionOptions()
		stat = TF_NewStatus()

		session = TF_LoadSessionFromSavedModel(sessionoptions, &
			model_dir, &
			tags, size(tags, 1), graph, stat)

		if (TF_GetCode( stat ) .ne. TF_OK) then
			call TF_Message( stat, message )
			write(*,*) TF_GetCode( stat ), message
			stop
		endif

		call TF_DeleteSessionOptions(sessionoptions)
		call TF_DeleteStatus(stat)

	end subroutine load_model


	subroutine {{ module_name }}_calc( &
		session, inputs, input_tensors, outputs, output_tensors &
	)

		type(TF_Session) :: session
		type(TF_Output), dimension(:) :: inputs, outputs
		type(TF_Tensor), dimension(:) :: input_tensors, output_tensors

		type(TF_Status) :: stat
		character(100) :: message
		type(TF_Operation), dimension(1) :: target_opers

		stat = TF_NewStatus()

		call TF_SessionRun( &
			session, &
			inputs, input_tensors, &
			size(input_tensors), &
			outputs, output_tensors, &
			size(output_tensors), &
			target_opers, 0, stat &
		)
		if (TF_GetCode(stat) .ne. TF_OK) then
			call TF_Message(stat, message)
			write(*,*) TF_GetCode(stat), message
			stop
		endif
		call TF_DeleteStatus(stat)

	end subroutine {{ module_name }}_calc

	subroutine {{ module_name }}_finish()

		type(TF_Status) :: stat
		character(100) :: message

		stat = TF_NewStatus()
		! Delete the model variables.
        {% for model in models %}
		! Model: {{ model['model_dir'] }}
		call TF_DeleteGraph(model_graph_{{ loop.index }})
		call TF_DeleteSession(model_session_{{ loop.index }}, &
			stat)
		if (TF_GetCode(stat) .ne. TF_OK) then
			call TF_Message(stat, message)
			write(*,*) TF_GetCode(stat), message
			! we don't stop here so all resources can try to delete
		endif
        {% endfor %}
		call TF_DeleteStatus(stat)

	end subroutine {{ module_name }}_finish


