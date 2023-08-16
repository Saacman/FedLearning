import subprocess
from resnet_quant_fedlern_script import main


reps = 3
bits = [3,4,5,6,7,8]

j = 0
for arguments in bits:
    for i in range(reps):

        main(i, arguments)
        j += 1
