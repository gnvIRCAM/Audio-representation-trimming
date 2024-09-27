DEVICE=""
DB_PATH=""

while [[ "$1" != "" ]]; do
    case $1 in 
    --device )
        shift
        DEVICE=$1
        ;;
    --db )
        shift
        DB_PATH=$1
        ;;
    * )
        usage
        ;;
    esac
    shift
done

alias train_commands="python train.py --config configs/models/foundation/wav2vec --config configs/task/classif --device $DEVICE --db $DB_PATH" 
train_commands --config configs/experiments/base.gin --run_name runs/wav2vec/commands/base  
train_commands --config configs/experiments/ssf.gin --run_name runs/wav2vec/commands/ssf
train_commands --config configs/experiments/mask.gin --run_name runs/wav2vec/commands/mask/sparsity_25    
train_commands --config configs/experiments/mask.gin --run_name runs/wav2vec/commands/mask/sparsity_50    
train_commands --config configs/experiments/mask.gin --run_name runs/wav2vec/commands/mask/sparsity_75   