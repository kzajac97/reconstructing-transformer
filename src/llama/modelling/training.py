def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def train(model, train_dataset, lora_config, data_collator, tokenizer, training_arguments):
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=lora_config,
        data_collator=data_collator,
        dataset_text_field="data",
        tokenizer=tokenizer,
        args=training_arguments,
    )

    trainer.train()
    return trainer, model
