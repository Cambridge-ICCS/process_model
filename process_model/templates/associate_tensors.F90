	function {{ prefix }}_{{ input_shape_len }}_associate_tensor(input_array, input_shape, input_size)
		type(TF_Tensor) :: {{ prefix }}_{{ input_shape_len }}_associate_tensor
		{{ input_type_dec }}, dimension({{ input_dimension_str }}), target :: input_array
		integer(kind=c_int64_t), dimension({{ input_shape_len }}), optional :: input_shape
		integer(kind=c_size_t), optional :: input_size

		integer(kind=c_int64_t), dimension({{ input_shape_len }}) :: input_shape_act
		integer(kind=c_int64_t) :: swap
		integer :: i, sz_inp_act
		integer(kind=c_size_t) :: input_size_act

		if (.not.present(input_shape)) then
			input_shape_act = shape(input_array)
		else
			input_shape_act = input_shape
		end if

		! Reverse the index order of the shape.
		sz_inp_act = size(input_shape_act) + 1 ! 1-indexed arrays
		do i = 1, sz_inp_act / 2
			swap = input_shape_act(i)
			input_shape_act(i) = input_shape_act(sz_inp_act - i)
			input_shape_act(sz_inp_act - i) = swap
		enddo

		if (.not.present(input_size)) then
			! sizeof is non-standard but seems to be widely supported.
			input_size_act = int(sizeof(input_array), kind=c_size_t)
		else
			input_size_act = input_size
		end if

		{{ prefix }}_{{ input_shape_len }}_associate_tensor = TF_NewTensor(TF_FLOAT, input_shape_act, {{ input_shape_len }}, &
			c_loc(input_array), input_size_act)

	end function {{ prefix }}_{{ input_shape_len }}_associate_tensor

