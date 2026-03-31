class _NoOpLogger:
    def log(self, *args, **kwargs):
        return None

    def finish(self):
        return None


def build_swanlab_logger(args):
    if not getattr(args, "use_swanlab", False):
        return _NoOpLogger()

    try:
        import swanlab
    except ImportError as exc:
        raise RuntimeError("SwanLab is enabled but not installed. Install it with `pip install swanlab`.") from exc

    if getattr(args, "swanlab_api_key", ""):
        swanlab.login(
            api_key=args.swanlab_api_key,
            save=getattr(args, "swanlab_save_key", False),
        )

    run = swanlab.init(
        project=args.swanlab_project,
        workspace=args.swanlab_workspace or None,
        experiment_name=args.swanlab_experiment_name or None,
        description=args.swanlab_description or None,
        config={k: v for k, v in vars(args).items() if not k.startswith("swanlab_")},
        mode=args.swanlab_mode,
        tags=getattr(args, "swanlab_tags", None),
    )

    class _SwanLabLogger:
        def __init__(self, module, current_run):
            self._module = module
            self._run = current_run

        def log(self, data, step=None):
            self._module.log(data, step=step)

        def finish(self):
            if self._run is not None:
                self._run.finish()

    return _SwanLabLogger(swanlab, run)
