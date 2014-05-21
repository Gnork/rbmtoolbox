function processData(in, labelsOut, namesOut)
	load (in)
	save (labelsOut, "S")
	save (namesOut, "names")
endfunction
