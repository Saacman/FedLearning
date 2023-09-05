#!/bin/bash

script_filename="resnet_quant_fedlern_script.py"
reps=3
bits=(3 4 5 6 7 8)

j=0
for b in "${bits[@]}"; do
        for ((i=0; i<$reps; i++)); do
                    python3 "$script_filename" "$j" "$b"
                            j=$((j+1))
                                done
                            done

