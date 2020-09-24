from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
from torch.utils.data import DataLoader
from l5kit.configs import load_config_data

cfg = load_config_data("../input/lyft-config-files/agent_motion_config.yaml")
train_cfg = cfg["train_data_loader"]
os.environ["L5KIT_DATA_FOLDER"] = "/kaggle/input/lyft-motion-prediction-autonomous-vehicles"
dm = LocalDataManager(None)
rasterizer = build_rasterizer(cfg, dm)
train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
train_dataloader = DataLoader(train_dataset,
                              shuffle=train_cfg["shuffle"],
                              batch_size=train_cfg["batch_size"],
                              num_workers=train_cfg["num_workers"])

valid_path = 'scenes/validate.zarr'
valid_zarr = ChunkedDataset(dm.require(valid_path)).open()
print("valid_zarr", type(train_zarr))
valid_dataset = AgentDataset(cfg, valid_zarr, rasterizer)
valid_loader = DataLoader(
    valid_dataset,
    shuffle=False,
    batch_size=12,
    num_workers=16
)
