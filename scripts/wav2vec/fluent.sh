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

alias train_fluent="python train.py --config configs/models/foundation/wav2vec --config configs/task/classif --device $DEVICE --db $DB_PATH" 
train_fluent --config configs/experiments/base.gin --run_name runs/wav2vec/fluent/base  
train_fluent --config configs/experiments/ssf.gin --run_name runs/wav2vec/fluent/ssf
train_fluent --config configs/experiments/mask.gin --run_name runs/wav2vec/fluent/mask/sparsity_25    
train_fluent --config configs/experiments/mask.gin --run_name runs/wav2vec/fluent/mask/sparsity_50    
train_fluent --config configs/experiments/mask.gin --run_name runs/wav2vec/fluent/mask/sparsity_75   