python -m src.main_scene +experiment=${config} \
  data_loader.train.num_workers=${num_workers} \
  mode=train \
  trainer.devices=${gpus} \
  trainer.num_nodes=${num_nodes} \
  wandb.activated=true

# # RealEstate10K
# scenegen_shift1_re10k   # timestep_shift=1
# scenegen_shift4_re10k   # timestep_shift=4
# scenegen_shift12_re10k  # timestep_shift=12