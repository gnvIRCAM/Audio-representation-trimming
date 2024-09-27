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

alias train_nsynth="python train.py --config configs/models/foundation/musicfm --config configs/task/classif --device $DEVICE --db $DB_PATH" 
train_nsynth --config configs/experiments/base.gin --run_name runs/musicfm/nsynth/base  
train_nsynth --config configs/experiments/ssf.gin --run_name runs/musicfm/nsynth/ssf
train_nsynth --config configs/experiments/mask.gin --run_name runs/musicfm/nsynth/mask/sparsity_25    
train_nsynth --config configs/experiments/mask.gin --run_name runs/musicfm/nsynth/mask/sparsity_50    
train_nsynth --config configs/experiments/mask.gin --run_name runs/musicfm/nsynth/mask/sparsity_75    