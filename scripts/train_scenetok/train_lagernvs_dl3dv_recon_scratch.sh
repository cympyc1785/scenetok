config=custom/lagernvs_va-wan_dl3dv_recon_scratch
num_workers=8
gpus=1
num_nodes=1
exp_name="lagernvs_va-wan_dl3dv_recon_scratch"


# WANDB_API_KEY 는 커밋 파일에 하드코딩 금지 — 실행 전 env 에 키 export.
#   export WANDB_API_KEY=<key>
: "${WANDB_API_KEY:?set WANDB_API_KEY in env before running}"
export DEBUG=1

CUDA_VISIBLE_DEVICES=2 python -m src.main +experiment=${config} \
  data_loader.train.num_workers=${num_workers} \
  mode=train \
  trainer.devices=${gpus} \
  trainer.num_nodes=${num_nodes} \
  trainer.num_sanity_val_steps=1 \
  hydra.run.dir=exp/${exp_name} \
  checkpointing.dirpath=my_checkpoints/${exp_name} \
  wandb.activated=true \
  hydra.run.dir=exp/${exp_name} \
  checkpointing.dirpath=my_checkpoints/${exp_name}
