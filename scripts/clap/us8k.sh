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

alias train_us8k="python train.py --config configs/models/foundation/clap --config configs/task/classif --device $DEVICE --db $DB_PATH" 
train_us8k --config configs/experiments/base.gin --run_name runs/clap/us8k/base  
train_us8k --config configs/experiments/ssf.gin --run_name runs/clap/us8k/ssf
train_us8k --config configs/experiments/mask.gin --run_name runs/clap/us8k/mask/sparsity_25    
train_us8k --config configs/experiments/mask.gin --run_name runs/clap/us8k/mask/sparsity_50    
train_us8k --config configs/experiments/mask.gin --run_name runs/clap/us8k/mask/sparsity_75   