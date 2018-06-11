
num_worker=2

MODEL_DIR=`pwd`/model_dir
chief_lock=${MODEL_DIR}/chief.lock
rm -rf ${chief_lock}

# start chief
export TF_CONFIG='{
    "cluster": {
        "chief": ["localhost:2222"],
        "worker": ["localhost:2223", "localhost:2224"],
        "ps": ["localhost:2225"]
    },
    "task": {"type": "chief", "index": 0}
}'

echo "start chief ..."
bash ./train.sh > chief.log 2>&1 &


for ((i=0; i < num_worker; i++))
do
    export TF_CONFIG='{
        "cluster": {
            "chief": ["localhost:2222"],
            "worker": ["localhost:2223", "localhost:2224"],
            "ps": ["localhost:2225"]
        },
        "task": {"type": "worker", "index": '${i}'}
    }'
    echo "start worker $i ..."
    bash ./train.sh > worker.${i}.log 2>&1 &
done

# start ps
export TF_CONFIG='{
    "cluster": {
        "chief": ["localhost:2222"],
        "worker": ["localhost:2223", "localhost:2224"],
        "ps": ["localhost:2225"]
    },
    "task": {"type": "ps", "index": 0}
}'

echo "start ps ..."
bash ./train.sh > ps.log 2>&1 &


# start evaluator
export TF_CONFIG='{
    "cluster": {
        "chief": ["localhost:2222"],
        "worker": ["localhost:2223", "localhost:2224"],
        "ps": ["localhost:2225"]
    },
    "task": {"type": "evaluator", "index": 0}
}'

echo "start evaluator ..."
bash ./train.sh > evaluator.log 2>&1 &
