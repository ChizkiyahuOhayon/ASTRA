# Shared settings for the benchmark scripts. Source it; do not run it.
#
# METHOD=astra (default) trains ASTRA; METHOD=adgs trains the unmodified AD-GS baseline
# with the same schedule, which is how every "AD-GS (reproduced)" row was produced.
METHOD=${METHOD:-astra}

# Absolute-gradient densification (custom rasterizer) + anisotropy regulariser.
BACKGROUND=(--abs_grad --densify_scene_grad_threshold 0.00072
            --densify_obj_grad_threshold 0.00072 --lambda_aniso 0.003)
# Rigid tracklet motion for tracklets seen in >= 8 frames.
OBJECTS=(--shared_motion --shared_motion_support_gate --shared_motion_basis_cache)
# Evidence routing: points without enough support keep the AD-GS motion basis.
ROUTING=(--shared_motion_adgs_ungated)

# train_and_render <config> <source> <model_dir> <iterations> [extra train args...]
train_and_render() {
    local config=$1 source=$2 model=$3 steps=$4; shift 4
    local flags=() rasterizer=default
    if [[ $METHOD == astra ]]; then flags=("${ASTRA_FLAGS[@]}"); rasterizer=abs; fi
    if [[ -s $model/results.json ]]; then echo "skip $model (done)"; return; fi
    mkdir -p "$model"
    echo "[$(date +%H:%M)] $METHOD  $source -> $model"
    ADGS_RASTERIZER=$rasterizer python train.py -c "$config" -s "$source" -m "$model" \
        --iterations "$steps" --save_iterations "$steps" --test_iterations "$steps" \
        "$@" "${flags[@]}" > "$model/train.log" 2>&1
    ADGS_RASTERIZER=$rasterizer python render.py -c "$config" -m "$model" \
        --iteration "$steps" --skip_train > "$model/render.log" 2>&1
}
