import pandas as pd
import pytest

try:  # pragma: no cover - environment dependent
    import lightgbm  # noqa: F401
except Exception:
    pytest.skip("LightGBM not available", allow_module_level=True)

from matcher.ranking.trainer import predict_scores, train_lambdarank


def _make_feature_table() -> pd.DataFrame:
    rows = []
    for vacancy_index in range(5):
        vacancy_id = f"vac{vacancy_index}"
        rows.append(
            {
                "idVacancy": vacancy_id,
                "idCv": f"{vacancy_id}_a",
                "relevance": 1,
                "skill_overlap": 3.0,
                "skill_precision": 0.6,
                "skill_recall": 0.5,
                "skill_f1": 0.55,
                "requirements_coverage": 0.8,
                "bm25_score": 2.0,
                "resume_token_len": 120,
                "job_token_len": 80,
                "resume_experience_count": 2,
                "resume_education_count": 1,
            }
        )
        rows.append(
            {
                "idVacancy": vacancy_id,
                "idCv": f"{vacancy_id}_b",
                "relevance": 0,
                "skill_overlap": 1.0,
                "skill_precision": 0.2,
                "skill_recall": 0.15,
                "skill_f1": 0.17,
                "requirements_coverage": 0.2,
                "bm25_score": 0.5,
                "resume_token_len": 100,
                "job_token_len": 80,
                "resume_experience_count": 1,
                "resume_education_count": 1,
            }
        )
    return pd.DataFrame(rows)


def test_train_lambdarank_returns_model() -> None:
    table = _make_feature_table()
    artifacts = train_lambdarank(
        table,
        validation_frac=0.2,
        test_frac=0.2,
        params={"num_leaves": 15, "min_data_in_leaf": 1},
        num_boost_round=20,
        early_stopping_rounds=5,
    )
    assert artifacts.model is not None
    assert "ndcg@3" in artifacts.validation_metrics


def test_predict_scores_shapes_match() -> None:
    table = _make_feature_table()
    artifacts = train_lambdarank(
        table,
        validation_frac=0.2,
        test_frac=0.2,
        params={"num_leaves": 15, "min_data_in_leaf": 1},
        num_boost_round=10,
        early_stopping_rounds=3,
    )
    preds = predict_scores(artifacts.model, table)
    assert preds.shape[0] == len(table)
