"""Build the environment for a network name."""

from .env import EnvConfig, TSCEnv

NETWORKS = ("1tls_3x3", "cologne1", "ingolstadt1")


def make_env(cfg: EnvConfig, net: str = "1tls_3x3"):
    if net == "1tls_3x3":
        return TSCEnv(cfg)
    from .realnet import RealNetEnv
    return RealNetEnv(cfg, net)
