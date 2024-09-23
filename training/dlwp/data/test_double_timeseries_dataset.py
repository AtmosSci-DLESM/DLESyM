from training.dlwp.data.data_loading import DoubleTimeSeriesDataset
from omegaconf import OmegaConf, DictConfig
import hydra



# executable block foretesting 
@hydra.main(version_base=None,config_path='../../configs', config_name="config_CoupledUnet")
def instantiate_model(cfg: DictConfig) -> None:

    print(OmegaConf.to_yaml(cfg.data))
    data = hydra.utils.instantiate(cfg.data)

if __name__ == "__main__" :

    instantiate_model()
