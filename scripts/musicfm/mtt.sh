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

alias train_mtt="python train.py --config configs/models/foundation/musicfm --config configs/task/tagging --device $DEVICE --db $DB_PATH" 
train_mtt --config configs/experiments/base.gin --run_name runs/musicfm/mtt/base  
train_mtt --config configs/experiments/ssf.gin --run_name runs/musicfm/mtt/ssf
train_mtt --config configs/experiments/mask.gin --run_name runs/musicfm/mtt/mask/sparsity_25    
train_mtt --config configs/experiments/mask.gin --run_name runs/musicfm/mtt/mask/sparsity_50    
train_mtt --config configs/experiments/mask.gin --run_name runs/musicfm/mtt/mask/sparsity_75    