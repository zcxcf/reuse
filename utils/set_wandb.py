import wandb


def set_wandb(args):
    wandb.init(
        project="reuse",
        config={
            "learning_rate": args.blr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        },
        name=args.model+args.initial
    )

