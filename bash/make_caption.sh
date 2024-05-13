set -x
export INPUT_DIR=$1;
export PNG_NAME=$2;
export OUTPUT_DIR=$3;
export SH_DIR=$4;
export SUBJECT_NAME=$(basename $1 | cut -d"." -f1);
export REPLICATE_API_TOKEN="r8_R6tI0diNzRXbD6sCovQluVS6vEbTgps35uaH7"; # your replicate token for BLIP API
export CUDA_HOME=/usr/local/cuda-11.3/ #/your/cuda/home/dir;
export PYOPENGL_PLATFORM=osmesa
export MESA_GL_VERSION_OVERRIDE=4.1
export PYTHONPATH=$PYTHONPATH:$(pwd);

# Step 2: Get BLIP prompt and gender, you can also use your own prompt
CUDA_VISIBLE_DEVICES=0 python scripts/get_prompt_blip.py --img-path ${INPUT_DIR}/${PNG_NAME}.png  --out-path ${INPUT_DIR}/prompt.txt

export PROMPT="$(cat ${INPUT_DIR}/prompt.txt| cut -d'|' -f1)"
export GENDER=`cat ${INPUT_DIR}/prompt.txt| cut -d'|' -f2`

python scripts/get_sh.py --input_dir ${INPUT_DIR} \
                         --output_dir ${OUTPUT_DIR} \
                         --origin_image_path ${INPUT_DIR}/${PNG_NAME}.png \
                         --obj_name mesh_normalized \
                         --obj_file mesh_normalized.obj \
                         --prompt_file_path ${INPUT_DIR}/prompt.txt \
                         --output_sh_path ./bash/${SH_DIR}