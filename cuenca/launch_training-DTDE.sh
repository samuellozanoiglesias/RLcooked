#!/bin/bash

# Forzar el uso de punto decimal en printf
LC_NUMERIC=en_US.UTF-8

# Argumentos para el script Python
cluster=cuenca
input=input_0.4_0.2
input_path=$input.txt
map_nr=baseline_division_of_labor_large
lr=0.0003
game_version=classic
num_epochs=20
num_agents=1
seeds=(0)  # Lista de seeds para ejecutar
checkpoint_paths=none
rewards_only_on_delivery=false
random_initial_state=true
ability_risk_enabled=false
eta=0.0
specialization_penalty_enabled=false
agent_to_train=2 

# Iterar sobre cada seed
for seed in "${seeds[@]}"; do
    output_file=output_$map_nr-$num_agents-$game_version-$input-seed_$seed_$random_initial_state.txt
    
    # Ejecutar el entrenamiento en segundo plano con nohup y argumentos
    nohup python /home/samuel_lozano/cooked/training-DTDE-spoiled_broth.py $cluster $input_path $map_nr $lr $game_version $num_agents $num_epochs $seed $checkpoint_paths $rewards_only_on_delivery $random_initial_state $ability_risk_enabled $eta $specialization_penalty_enabled $agent_to_train > $output_file 2>&1 &
    
    echo "Lanzado entrenamiento con seed=$seed -> argumentos: $cluster $input_path $map_nr $lr $game_version $num_agents $num_epochs $seed $checkpoint_paths $rewards_only_on_delivery $random_initial_state $ability_risk_enabled $eta $specialization_penalty_enabled $agent_to_train"
    
    # Esperar 45 segundos antes de lanzar la siguiente simulación
    if [ "$seed" != "${seeds[-1]}" ]; then
        echo "Esperando 45 segundos antes de la siguiente simulación..."
        sleep 45
    fi
done

echo "Todas las simulaciones han sido lanzadas."