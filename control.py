import subprocess
from resnet_quant_fedlern_script import main


reps = 1
bits = [3,9,10,11,12,13,14,15,16]

j = 0
for arguments in bits:
    for i in range(reps):

        main(j, arguments)
        j += 1
