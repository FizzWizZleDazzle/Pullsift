"""Shared pieces of the self-hosted AI-text detector: prose extraction
and model loading. Used by detector_server.py (live scoring) and
detector_batch.py (corpus scoring).

The model is desklib/ai-text-detector-v1.01 (DeBERTa-v3-large, MIT), the
strongest permissively-licensed open model on the RAID benchmark. Scores
are a weak signal by design: the engine gives them one fitted weight and
they cannot reach an enforcement tier alone.
"""

import re

MIN_WORDS = 50  # below this, detectors are noise; abstain

CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE = re.compile(r"`[^`]+`")
URL = re.compile(r"https?://\S+")
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
CHECKBOX = re.compile(r"^\s*-\s*\[[ xX]\]\s*", re.MULTILINE)
HEADING = re.compile(r"^#{1,6}\s+", re.MULTILINE)
DIFF_LINE = re.compile(r"^[+-]{1,3}\s?.*$", re.MULTILINE)


def extract_prose(text: str) -> str:
    """Strip everything detectors are not trained on: code, links,
    template scaffolding, diff fragments."""
    t = CODE_FENCE.sub(" ", text)
    t = HTML_COMMENT.sub(" ", t)
    t = INLINE_CODE.sub(" ", t)
    t = URL.sub(" ", t)
    t = CHECKBOX.sub("", t)
    t = HEADING.sub("", t)
    return re.sub(r"\s+", " ", t).strip()


def usable(prose: str) -> bool:
    return len(prose.split()) >= MIN_WORDS


def load_model(model_id="desklib/ai-text-detector-v1.01"):
    """Load the detector. Returns (tokenizer, model, score_fn)."""
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel

    class DesklibModel(PreTrainedModel):
        config_class = AutoConfig

        def __init__(self, config):
            super().__init__(config)
            self.model = AutoModel.from_config(config)
            self.classifier = torch.nn.Linear(config.hidden_size, 1)
            self.init_weights()

        def forward(self, input_ids, attention_mask=None):
            out = self.model(input_ids, attention_mask=attention_mask)
            hidden = out[0]
            mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
            pooled = torch.sum(hidden * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
            return self.classifier(pooled)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = DesklibModel.from_pretrained(model_id)
    model.eval()

    def score(text: str):
        import torch

        enc = tokenizer(
            text, truncation=True, max_length=768, padding=True, return_tensors="pt"
        )
        with torch.no_grad():
            logits = model(enc["input_ids"], attention_mask=enc["attention_mask"])
            return torch.sigmoid(logits).item()

    return tokenizer, model, score
