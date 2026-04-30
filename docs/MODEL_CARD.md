# Model Card

## Model

Support Bot v1 BERT classifier.

Base model:

```text
ytu-ce-cosmos/turkish-base-bert-uncased
```

Task:

```text
Turkish support message category classification
```

## Classes

The model predicts 24 support categories. See `artifacts/tr_bert_uncased_epoch9/label_mapping.json` for the exact id-to-label mapping.

## Training Data

Training file:

```text
data_v6.xlsx
```

Columns:

- `kullanici_mesaji`: original user message
- `rewrite`: normalized/re-written user message
- `kategori`: target label

Rows used after preparation:

```text
6928
```

## Metrics

Validation split:

```text
15%
```

Metrics:

```text
Accuracy: 0.8384615385
Macro F1: 0.8230398204
```

## Size

Main weight file:

```text
model.safetensors: ~422 MB
```

Inference folder with tokenizer/config:

```text
bert_model/: ~423 MB
```

Full training artifacts with checkpoints:

```text
~4.13 GB
```

## Intended Use

The model is intended for routing and assisting support messages in a Turkish online game/support domain. It should be used as a support-assistance component, not as the final authority for sensitive account or payment decisions.

## Limitations

- Performance depends on similarity between production messages and `data_v6.xlsx`.
- Very short or ambiguous messages may need human review.
- Rewrite can help normalize noisy messages, but it can also introduce errors if the local LLM misunderstands the domain.

