set -x

MODEL_PATH=microsoft/Magma-8B  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config_magma.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=magma_8b_geo_grpo \
    trainer.n_gpus_per_node=4