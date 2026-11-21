import torch
from torch.utils.data import DataLoader
from transformers import BertForTokenClassification


def largest(scores, size):
    return scores.topk(size).indices.sort().values


def copy_linear(dst, src, rows, cols):
    dst.weight.copy_(src.weight[rows][:, cols])
    dst.bias.copy_(src.bias[rows])


def copy_norm(dst, src, keep):
    dst.weight.copy_(src.weight[keep])
    dst.bias.copy_(src.bias[keep])


def prune_student(teacher, config, blocks, sample, collator):
    """Выбирает компоненты по активациям teacher и переносит их веса в student."""
    hidden_scores = torch.zeros(768, device=teacher.device)
    head_scores = {i: torch.zeros(12, device=teacher.device) for i in blocks}
    ffn_scores = {i: torch.zeros(3072, device=teacher.device) for i in blocks}

    def hidden_hook(module, inputs, output):
        hidden_scores.add_(output[mask].float().abs().sum(dim=0))

    def head_hook(i):
        def hook(module, inputs):
            values = inputs[0][mask].float().reshape(-1, 12, 64)
            head_scores[i].add_(values.norm(dim=-1).sum(dim=0))

        return hook

    def ffn_hook(i):
        def hook(module, inputs):
            ffn_scores[i].add_(inputs[0][mask].float().abs().sum(dim=0))

        return hook

    hooks = [teacher.bert.embeddings.LayerNorm.register_forward_hook(hidden_hook)]

    for i in blocks:
        layer = teacher.bert.encoder.layer[i]
        hooks.append(layer.attention.output.LayerNorm.register_forward_hook(hidden_hook))
        hooks.append(layer.output.LayerNorm.register_forward_hook(hidden_hook))
        hooks.append(layer.attention.output.dense.register_forward_pre_hook(head_hook(i)))
        hooks.append(layer.output.dense.register_forward_pre_hook(ffn_hook(i)))

    loader = DataLoader(sample, batch_size=8, collate_fn=collator)

    try:
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(teacher.device) for k, v in batch.items() if k != "labels"}
                mask = batch["attention_mask"].bool()
                teacher(**batch)
    finally:
        for hook in hooks:
            hook.remove()

    student = BertForTokenClassification(config)
    student.to(teacher.device)
    keep = largest(hidden_scores, 256)

    with torch.no_grad():
        src = teacher.bert.embeddings
        dst = student.bert.embeddings

        for name in ["word_embeddings", "position_embeddings", "token_type_embeddings"]:
            getattr(dst, name).weight.copy_(getattr(src, name).weight[:, keep])

        copy_norm(dst.LayerNorm, src.LayerNorm, keep)

        for i, j in enumerate(blocks):
            src = teacher.bert.encoder.layer[j]
            dst = student.bert.encoder.layer[i]

            heads = largest(head_scores[j], 4)
            cols = (heads[:, None] * 64 + torch.arange(64, device=teacher.device)).flatten()

            for name in ["query", "key", "value"]:
                copy_linear(
                    getattr(dst.attention.self, name),
                    getattr(src.attention.self, name),
                    cols,
                    keep
                )

            copy_linear(
                dst.attention.output.dense,
                src.attention.output.dense,
                keep,
                cols
            )
            copy_norm(
                dst.attention.output.LayerNorm,
                src.attention.output.LayerNorm,
                keep
            )

            neurons = largest(ffn_scores[j], 1024)
            copy_linear(dst.intermediate.dense, src.intermediate.dense, neurons, keep)
            copy_linear(dst.output.dense, src.output.dense, keep, neurons)
            copy_norm(dst.output.LayerNorm, src.output.LayerNorm, keep)

        labels = torch.arange(student.config.num_labels, device=teacher.device)
        copy_linear(student.classifier, teacher.classifier, labels, keep)

    return student
