
model_dir=$(pwd)/export_model_dir
latest_model=${model_dir}/`ls ${model_dir} | sort | tail -n1`
echo -e "latest_model path: " ${latest_model} "\n"

saved_model_cli show --dir ${latest_model} --tag_set serve --all
