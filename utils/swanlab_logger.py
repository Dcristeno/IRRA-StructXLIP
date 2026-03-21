import os

from utils.comm import get_rank


class SwanLabLogger:
    def __init__(self, enabled=False, run=None, module=None):
        self.enabled = enabled
        self.run = run
        self.module = module

    def log(self, data, step=None):
        if not self.enabled:
            return
        if self.run is not None and hasattr(self.run, "log"):
            if step is None:
                self.run.log(data)
            else:
                self.run.log(data, step=step)
            return
        if self.module is None:
            return
        if step is None:
            self.module.log(data)
        else:
            self.module.log(data, step=step)

    def finish(self):
        if not self.enabled:
            return
        if self.run is not None and hasattr(self.run, "finish"):
            self.run.finish()
            return
        if self.module is not None and hasattr(self.module, "finish"):
            self.module.finish()


def _sanitize_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [_sanitize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _sanitize_value(v) for k, v in value.items()}
    return str(value)


def build_swanlab_logger(args):
    if not getattr(args, "use_swanlab", False):
        return SwanLabLogger(enabled=False)

    if get_rank() != 0:
        return SwanLabLogger(enabled=False)

    try:
        import swanlab
    except ImportError as exc:
        raise RuntimeError(
            "SwanLab is enabled but not installed. Please run `pip install swanlab`."
        ) from exc

    api_key = getattr(args, "swanlab_api_key", "") or os.getenv("SWANLAB_API_KEY")
    if api_key:
        swanlab.login(api_key=api_key, save=getattr(args, "swanlab_save_key", False))

    config = {k: _sanitize_value(v) for k, v in vars(args).items() if k != "use_swanlab"}
    run = swanlab.init(
        project=args.swanlab_project,
        workspace=args.swanlab_workspace,
        experiment_name=args.swanlab_experiment_name if args.swanlab_experiment_name else args.name,
        description=args.swanlab_description,
        group=args.dataset_name,
        tags=args.swanlab_tags,
        config=config,
        logdir=os.path.join(args.output_dir, "swanlog"),
        mode=args.swanlab_mode,
    )
    return SwanLabLogger(enabled=True, run=run, module=swanlab)
