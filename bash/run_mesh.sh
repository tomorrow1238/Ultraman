set -x
export INPUT_PNG_DIR=$1;
export PNG_NAME=$2
export FINAL_RESULT_PATH=$3


python models/2K2K/test_02_model.py --data_path ${INPUT_PNG_DIR} --checkpoints_load_path ./models/2K2K/checkpoints/ --save_path ${INPUT_PNG_DIR}
python models/2K2K/test_03_poisson.py --save_path ${INPUT_PNG_DIR}

mkdir -p ${FINAL_RESULT_PATH}

mv ${INPUT_PNG_DIR}/${PNG_NAME}.png ${FINAL_RESULT_PATH}

python scripts/decimation_mesh.py --input_path ${INPUT_PNG_DIR}/test/output_objs/${PNG_NAME}.obj --output_path ${FINAL_RESULT_PATH}
python scripts/generate_uv.py --input_dir ${FINAL_RESULT_PATH} --obj_name decimation_mesh --output_dir ${FINAL_RESULT_PATH}
python scripts/normalize_mesh.py --input_dir ${FINAL_RESULT_PATH} --obj_name mesh

