import torch
from torch import nn
from transformers import Trainer

class HiddenTrainer(Trainer):
    def __init__(self, teacher, temperature=2.0, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.teacher = teacher.to(self.args.device)
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        self.temperature = temperature
        self.alpha = alpha
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.mse = nn.MSELoss()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs, output_hidden_states=True)
        teacher_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        with torch.no_grad():
            target = self.teacher(**teacher_inputs, output_hidden_states=True)

        mask = inputs["labels"] != -100
        t = self.temperature
        student_logits = outputs.logits[mask] / t
        teacher_logits = target.logits[mask] / t
        soft_loss = self.kl(student_logits.log_softmax(dim=-1), teacher_logits.softmax(dim=-1))

        hidden_loss = 0
        n = len(outputs.hidden_states) - 1
        m = len(target.hidden_states) - 1
        for i in range(1, n + 1):
            j = i * m // n
            hidden = model.projection(outputs.hidden_states[i][mask])
            hidden_loss += self.mse(hidden, target.hidden_states[j][mask])
        hidden_loss /= n
        loss = outputs.loss + t**2 * soft_loss + self.alpha * hidden_loss

        if return_outputs:
            return loss, {"logits": outputs.logits}
        return loss
