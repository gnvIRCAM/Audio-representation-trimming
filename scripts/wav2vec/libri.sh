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

alias train_libri="python train.py --config configs/models/foundation/wav2vec --config configs/task/classif --device $DEVICE --db $DB_PATH" 
train_libri --config configs/experiments/base.gin --run_name runs/wav2vec/libri/base  
train_libri --config configs/experiments/ssf.gin --run_name runs/wav2vec/libri/ssf
train_libri --config configs/experiments/mask.gin --run_name runs/wav2vec/libri/mask/sparsity_25    
train_libri --config configs/experiments/mask.gin --run_name runs/wav2vec/libri/mask/sparsity_50    
train_libri --config configs/experiments/mask.gin --run_name runs/wav2vec/libri/mask/sparsity_75   