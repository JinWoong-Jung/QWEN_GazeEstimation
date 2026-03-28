from __future__ import annotations


BATCH_LOCAL_INFONCE_OBJECTIVES = {
    "batch_local_infonce",
}

# NOTE:
# "infonce" is treated as full-vocab contrastive classification so that
# train-time objective matches test-time prediction space (pred_emb @ vocab_emb^T).
FULL_VOCAB_CONTRASTIVE_OBJECTIVES = {
    "infonce",
    "full_vocab_infonce",
    "prototype_ce",
    "vocab_ce",
}

EMBEDDING_RECOGNITION_OBJECTIVES = (
    BATCH_LOCAL_INFONCE_OBJECTIVES | FULL_VOCAB_CONTRASTIVE_OBJECTIVES
)


def normalize_recognition_objective(obj: str | None) -> str:
    return str(obj or "").strip().lower()


def is_batch_local_infonce_objective(obj: str | None) -> bool:
    return normalize_recognition_objective(obj) in BATCH_LOCAL_INFONCE_OBJECTIVES


def is_full_vocab_contrastive_objective(obj: str | None) -> bool:
    return normalize_recognition_objective(obj) in FULL_VOCAB_CONTRASTIVE_OBJECTIVES


def is_embedding_recognition_objective(obj: str | None) -> bool:
    return normalize_recognition_objective(obj) in EMBEDDING_RECOGNITION_OBJECTIVES
