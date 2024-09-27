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

alias train_fsd="python train.py --config configs/models/foundation/clap --config configs/task/tagging --device $DEVICE --db $DB_PATH" 
train_fsd --config configs/experiments/base.gin --run_name runs/clap/fsd/base  
train_fsd --config configs/experiments/ssf.gin --run_name runs/clap/fsd/ssf
train_fsd --config configs/experiments/mask.gin --run_name runs/clap/fsd/mask/sparsity_25    
train_fsd --config configs/experiments/mask.gin --run_name runs/clap/fsd/mask/sparsity_50    
train_fsd --config configs/experiments/mask.gin --run_name runs/clap/fsd/mask/sparsity_75   