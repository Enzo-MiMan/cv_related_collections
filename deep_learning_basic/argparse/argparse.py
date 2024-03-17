import torch
import datetime
import argparse


def diffusion_defaults():
    defaults = dict(
        num_timesteps=1000,
        schedule="linear",
        loss_type="l2",
        use_labels=False,

        base_channels=128,
        channel_mults=(1, 2, 2, 2),
        num_res_blocks=2,
        time_emb_dim=128 * 4,
        norm="gn",
        dropout=0.1,
        activation="silu",
        attention_resolutions=(1,),

        ema_decay=0.9999,
        ema_update_rate=1,
    )
    return defaults


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        parser.add_argument(f"--{k}", default=v, type=v_type)


def create_argparser():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    run_name = datetime.datetime.now().strftime("ddpm-%Y-%m-%d-%H-%M")
    defaults = dict(
        learning_rate=2e-4,
        batch_size=2,
        iterations=800000,

        log_to_wandb=True,
        log_rate=1000,
        checkpoint_rate=1000,
        log_dir="~/ddpm_logs",
        project_name="Enzo_ddpm",
        run_name=run_name,

        model_checkpoint=None,
        optim_checkpoint=None,

        schedule_low=1e-4,
        schedule_high=0.02,

        device=device,
    )
    defaults.update(diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == '__main__':
    args = create_argparser().parse_args()
    print(args)