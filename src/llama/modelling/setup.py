def setup():
    """Setup the training arguments and the model configuration for the LORA model."""
    lora_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="SEQ_CLS",
    )

    training_arguments = TrainingArguments(
        output_dir="training",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=1,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="wandb",
        push_to_hub=False,
        gradient_checkpointing=True,  # needs to be True: https://huggingface.co/docs/transformers/v4.18.0/en/performance
    )

    return lora_config, training_arguments
