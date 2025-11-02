import pandas as pd

from matcher.ranking.dataset_builder import build_feature_table


def test_build_feature_table_on_small_sample() -> None:
    data = [
        {
            "idCv": "cv1",
            "idVacancy": "vac1",
            "cv_status": "Приглашение",
            "skills_cv": '["Python", "SQL"]',
            "workExperienceList": '[{"jobTitle": "Python dev", "demands": "Разработка API"}]',
            "educationList": '[{"instituteName": "СПбГУ", "educationLevel": "Бакалавр"}]',
            "positionName": "Python разработчик",
            "vacancyName": "Python разработчик",
            "positionRequirements": "Python, SQL",
            "responsibilities": "Разработка API",
            "conditions": "Удаленная работа",
        },
        {
            "idCv": "cv2",
            "idVacancy": "vac1",
            "cv_status": "Отказ",
            "skills_cv": '["Java"]',
            "workExperienceList": '[{"jobTitle": "Java dev"}]',
            "educationList": "[]",
            "positionName": "Java разработчик",
            "vacancyName": "Python разработчик",
            "positionRequirements": "Python, SQL",
            "responsibilities": "Разработка API",
            "conditions": "Удаленная работа",
        },
    ]
    frame = pd.DataFrame(data)
    table = build_feature_table(frame)
    assert not table.empty
    assert {"idCv", "idVacancy", "relevance", "skill_overlap"} <= set(table.columns)
    assert table["relevance"].sum() == 1
