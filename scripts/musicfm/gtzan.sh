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

alias train_gtzan="python train.py --config configs/models/foundation/musicfm --config configs/task/classif --device $DEVICE --db $DB_PATH" 
train_gtzan --config configs/experiments/base.gin --run_name runs/musicfm/gtzan/base  
train_gtzan --config configs/experiments/ssf.gin --run_name runs/musicfm/gtzan/ssf
train_gtzan --config configs/experiments/mask.gin --run_name runs/musicfm/gtzan/mask/sparsity_25    
train_gtzan --config configs/experiments/mask.gin --run_name runs/musicfm/gtzan/mask/sparsity_50    
train_gtzan --config configs/experiments/mask.gin --run_name runs/musicfm/gtzan/mask/sparsity_75    