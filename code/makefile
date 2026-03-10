SSH_ADDR=hyan84@ap2001.chtc.wisc.edu

ssh_chtc:
	ssh hyan84@ap2001.chtc.wisc.edu

pass_file:
	scp $(MYFILE) $(SSH_ADDR):/home/hyan84/NWE

move_file:
	scp $(SSH_ADDR):/home/hyan84/NWE/$(FILE) .
