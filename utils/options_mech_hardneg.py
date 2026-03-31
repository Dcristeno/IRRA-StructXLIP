import argparse


def get_args_mech_hardneg():
    parser = argparse.ArgumentParser(description="IRRA Mechanism-Guided Hard Negative Args")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--name", default="mech-hardneg", help="experiment name to save")
    parser.add_argument("--output_dir", default="logs")
    parser.add_argument("--log_period", type=int, default=100)
    parser.add_argument("--eval_period", type=int, default=1)
    parser.add_argument("--val_dataset", default="test")
    parser.add_argument("--resume", default=False, action="store_true")
    parser.add_argument("--resume_ckpt_file", default="", help="resume from ...")

    parser.add_argument("--pretrain_choice", default="ViT-B/16")
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--img_aug", default=False, action="store_true")

    parser.add_argument("--cmt_depth", type=int, default=4)
    parser.add_argument("--masked_token_rate", type=float, default=0.8)
    parser.add_argument("--masked_token_unchanged_rate", type=float, default=0.1)
    parser.add_argument("--lr_factor", type=float, default=5.0)
    parser.add_argument("--MLM", default=False, action="store_true")

    parser.add_argument("--loss_names", default="sdm+id+mlm")
    parser.add_argument("--mlm_loss_weight", type=float, default=1.0)
    parser.add_argument("--id_loss_weight", type=float, default=1.0)

    parser.add_argument("--img_size", type=tuple, default=(384, 128))
    parser.add_argument("--stride_size", type=int, default=16)

    parser.add_argument("--text_length", type=int, default=77)
    parser.add_argument("--vocab_size", type=int, default=49408)

    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--bias_lr_factor", type=float, default=2.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--weight_decay_bias", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.999)

    parser.add_argument("--num_epoch", type=int, default=60)
    parser.add_argument("--milestones", type=int, nargs="+", default=(20, 50))
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--warmup_factor", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_method", type=str, default="linear")
    parser.add_argument("--lrscheduler", type=str, default="cosine")
    parser.add_argument("--target_lr", type=float, default=0)
    parser.add_argument("--power", type=float, default=0.9)

    parser.add_argument("--dataset_name", default="CUHK-PEDES")
    parser.add_argument("--sampler", default="random")
    parser.add_argument("--num_instance", type=int, default=4)
    parser.add_argument("--root_dir", default="./data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--test", dest="training", default=True, action="store_false")

    parser.add_argument("--dehn_loss_weight", type=float, default=1.5)
    parser.add_argument("--dehn_rank_loss_weight", type=float, default=1.0)
    parser.add_argument("--dehn_cov_loss_weight", type=float, default=0.5)
    parser.add_argument("--dehn_margin", type=float, default=0.18)
    parser.add_argument("--dehn_cov_margin", type=float, default=0.05)
    parser.add_argument("--dehn_neg_score_floor", type=float, default=0.25)
    parser.add_argument("--dehn_score_weight", type=float, default=3.0)
    parser.add_argument("--dehn_dominance_weight", type=float, default=4.0)
    parser.add_argument("--dehn_cov_weight", type=float, default=2.0)

    parser.add_argument("--use_swanlab", default=False, action="store_true")
    parser.add_argument("--swanlab_project", default="IRRA-StructXLIP")
    parser.add_argument("--swanlab_workspace", default="Dcristen")
    parser.add_argument("--swanlab_experiment_name", default="")
    parser.add_argument("--swanlab_description", default="")
    parser.add_argument("--swanlab_mode", default="cloud", choices=["cloud", "offline", "local", "disabled"])
    parser.add_argument("--swanlab_api_key", default="")
    parser.add_argument("--swanlab_save_key", default=False, action="store_true")
    parser.add_argument("--swanlab_tags", nargs="*", default=["irra", "mech-hardneg"])

    args = parser.parse_args()

    loss_names = [item.strip() for item in args.loss_names.split("+") if item.strip()]
    if "mlm" in loss_names and not args.MLM:
        print("`mlm` is included in `loss_names`, enabling `--MLM` automatically.")
        args.MLM = True

    return args
