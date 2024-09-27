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

alias train_esc="python train.py --config configs/models/foundation/clap --config configs/task/classif --device $DEVICE --db $DB_PATH" 
train_esc --config configs/experiments/base.gin --run_name runs/clap/esc/base  
train_esc --config configs/experiments/ssf.gin --run_name runs/clap/esc/ssf
train_esc --config configs/experiments/mask.gin --run_name runs/clap/esc/mask/sparsity_25    
train_esc --config configs/experiments/mask.gin --run_name runs/clap/esc/mask/sparsity_50    
train_esc --config configs/experiments/mask.gin --run_name runs/clap/esc/mask/sparsity_75   