class GPTConfig:
    """GPT model hyperparameters (~1.2M params)."""
    vocab_size = 4096
    max_seq_len = 256
    d_model = 256
    n_layers = 6
    n_heads = 8
    dropout = 0.0


class TrainConfig:
    """Training hyperparameters."""
    # data
    max_seq_len = GPTConfig.max_seq_len
    # optimization
    batch_size = 64
    learning_rate = 3e-4
    weight_decay = 0.1
    max_epochs = 5
    max_steps_per_epoch = 5000  # limit steps per epoch for streaming data
    # scheduler
    warmup_steps = 1000
    # mixed precision
    use_amp = True
    # logging
    log_interval = 100
    save_interval = 500
    eval_interval = 1000
    # generation sampling
    sample_interval = 1000
    sample_prompts = [
        "Once upon a time",
        "The little cat",
        "In a faraway land",
    ]


class FinetuneConfig:
    """Finetuning hyperparameters (lower LR, fewer epochs)."""
    # data
    max_seq_len = GPTConfig.max_seq_len
    # optimization
    batch_size = 64
    learning_rate = 3e-5
    weight_decay = 0.0
    max_epochs = 3
    max_steps_per_epoch = 2000
    # dropout (finetuning usually adds some)
    dropout = 0.1
    # mixed precision
    use_amp = True
    # logging
    log_interval = 100
    save_interval = 500
    # generation
    max_new_tokens = 100
