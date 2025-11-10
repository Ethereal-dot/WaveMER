from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.utilities.cli import LightningCLI

from WaveMER.datamodule import CROHMEDatamodule
from WaveMER.lit_wavemer import Lit_WaveMER

cli = LightningCLI(
    Lit_WaveMER,
    CROHMEDatamodule,
    save_config_overwrite=True,
    trainer_defaults={
        "plugins": DDPPlugin(find_unused_parameters=False),
        "gradient_clip_val": 1.0,   # ⭐ 开启梯度裁剪
    },
)
