
# start chief
export TF_CONFIG='{
    "cluster": {
        "chief": ["localhost:2222"],
        "worker": ["localhost:2223"],
        "ps": ["localhost:2224"]
    },
    "task": {"type": "chief", "index": 0}
}'

bash ./train.sh > chief.log 2>&1 &


# start worker
export TF_CONFIG='{
    "cluster": {
        "chief": ["localhost:2222"],
        "worker": ["localhost:2223"],
        "ps": ["localhost:2224"]
    },
    "task": {"type": "worker", "index": 0}
}'

bash ./train.sh > worker.log 2>&1 &


# start ps
export TF_CONFIG='{
    "cluster": {
        "chief": ["localhost:2222"],
        "worker": ["localhost:2223"],
        "ps": ["localhost:2224"]
    },
    "task": {"type": "ps", "index": 0}
}'

bash ./train.sh > ps.log 2>&1 &


# start evaluator
export TF_CONFIG='{
    "cluster": {
        "chief": ["localhost:2222"],
        "worker": ["localhost:2223"],
        "ps": ["localhost:2224"]
    },
    "task": {"type": "evaluator", "index": 0}
}'

bash ./train.sh > evaluator.log 2>&1 &
