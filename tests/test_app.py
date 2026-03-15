import pathlib
import importlib
import sys
from typing import Any

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier

@pytest.fixture()
def app_env(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> tuple[Any, TestClient]:
	if "app.app" in sys.modules:
		app_module = importlib.reload(sys.modules["app.app"])
	else:
		app_module = importlib.import_module("app.app")

	# Isolate DB and runtime globals so tests are deterministic.
	monkeypatch.setattr(app_module, "_DB_PATH", tmp_path / "patient_history_test.db")
	monkeypatch.setattr(app_module, "_SECRET_KEY", "unit-test-secret")
	monkeypatch.setattr(app_module, "_SESSION_DURATION", 3600)

	monkeypatch.setattr(
		app_module,
		"_feature_cols",
		[
			"Body_Temperature",
			"WBC_Count",
			"CRP",
			"Neutrophil_Percentage",
			"Appendix_Diameter",
			"Age",
			"Lower_Right_Abd_Pain",
			"Migratory_Pain",
			"Nausea",
			"Ipsilateral_Rebound_Tenderness",
		],
	)
	monkeypatch.setattr(
		app_module,
		"_defaults",
		{
			"Body_Temperature": 37.0,
			"WBC_Count": 10.0,
			"CRP": 5.0,
			"Neutrophil_Percentage": 70.0,
			"Appendix_Diameter": 6.0,
			"Age": 8.0,
			"Lower_Right_Abd_Pain": 0.0,
			"Migratory_Pain": 0.0,
			"Nausea": 0.0,
			"Ipsilateral_Rebound_Tenderness": 0.0,
		},
	)
	monkeypatch.setattr(app_module, "_model", object())
	monkeypatch.setattr(app_module, "predict_proba_safe", lambda _model, _x: 0.83)
	monkeypatch.setattr(app_module, "compute_shap_values", lambda _model, _x: ("sv", "base"))
	monkeypatch.setattr(app_module, "make_shap_waterfall_b64", lambda _sv, _base, _x: "fake-shap")

	return app_module, TestClient(app_module.app)


@pytest.fixture()
def client(app_env: tuple[Any, TestClient]) -> TestClient:
	return app_env[1]


def _login_as_admin(client: TestClient) -> None:
	app_module = importlib.import_module("app.app")
	response = client.post(
		"/login",
		data={"username": "admin", "password": "admin123"},
		follow_redirects=False,
	)
	assert response.status_code == 303
	assert app_module._SESSION_COOKIE in response.headers.get("set-cookie", "")


def _valid_predict_payload() -> dict[str, str]:
	return {
		"age": "11.2",
		"body_temperature": "38.3",
		"wbc_count": "14.0",
		"crp": "35.0",
		"neutrophil_percentage": "84.0",
		"appendix_diameter": "9.1",
		"lower_right_abd_pain": "yes",
		"migratory_pain": "yes",
		"nausea": "no",
		"ipsilateral_rebound_tenderness": "yes",
	}


def _fitted_dummy_estimator() -> DummyClassifier:
	X = np.array([[0.0], [1.0], [0.0], [1.0]])
	y = np.array([0, 1, 0, 1])
	model = DummyClassifier(strategy="most_frequent")
	model.fit(X, y)
	return model


def test_root_renders_landing_page(client: TestClient):
	response = client.get("/")
	assert response.status_code == 200
	assert "Aide au diagnostic" in response.text


def test_load_model_artifact_reads_valid_pkl(app_env: tuple[Any, TestClient], tmp_path: pathlib.Path):
	app_module, _ = app_env
	estimator = _fitted_dummy_estimator()
	model_path = tmp_path / "Random_Forest.pkl"
	joblib.dump(estimator, model_path)

	loaded = app_module._load_model_artifact(tmp_path)
	assert callable(getattr(loaded, "predict", None))
	assert callable(getattr(loaded, "predict_proba", None))


def test_load_model_artifact_reads_valid_joblib(app_env: tuple[Any, TestClient], tmp_path: pathlib.Path):
	app_module, _ = app_env
	estimator = _fitted_dummy_estimator()
	model_path = tmp_path / "random_forest.joblib"
	joblib.dump(estimator, model_path)

	loaded = app_module._load_model_artifact(tmp_path)
	assert callable(getattr(loaded, "predict", None))
	assert callable(getattr(loaded, "predict_proba", None))


def test_load_model_artifact_rejects_invalid_payload(app_env: tuple[Any, TestClient], tmp_path: pathlib.Path):
	app_module, _ = app_env
	joblib.dump({"name": "metadata-only"}, tmp_path / "random_forest.joblib")

	with pytest.raises(TypeError, match="non exploitables"):
		app_module._load_model_artifact(tmp_path)


def test_diagnosis_console_uses_api_predict_endpoint():
	tpl = pathlib.Path("app/templates/diagnosis_console.html").read_text(encoding="utf-8")
	assert "fetch('/api/predict'" in tpl
	assert "fetch('/predict'" not in tpl


def test_diagnosis_console_has_no_missing_js_symbols():
	tpl = pathlib.Path("app/templates/diagnosis_console.html").read_text(encoding="utf-8")
	assert "showDetailedResults(" not in tpl
	assert "tabResults" not in tpl
	assert "historyView" not in tpl
	assert "<parameter name=\"filePath\">" not in tpl


def test_build_input_row_orders_columns_and_encodes_binary(app_env: tuple[Any, TestClient]):
	app_module, _ = app_env
	form_data = {
		"age": "12",
		"body_temperature": "39.0",
		"wbc_count": "16.5",
		"crp": "20",
		"neutrophil_percentage": "88",
		"appendix_diameter": "10.2",
		"lower_right_abd_pain": "yes",
		"migratory_pain": "no",
		"nausea": "yes",
		"ipsilateral_rebound_tenderness": "no",
	}
	frame = app_module._build_input_row(form_data)

	assert list(frame.columns) == app_module._feature_cols
	row = frame.iloc[0].to_dict()
	assert row["Body_Temperature"] == 39.0
	assert row["WBC_Count"] == 16.5
	assert row["CRP"] == 20.0
	assert row["Neutrophil_Percentage"] == 88.0
	assert row["Appendix_Diameter"] == 10.2
	assert row["Age"] == 12.0
	assert row["Lower_Right_Abd_Pain"] == 1.0
	assert row["Migratory_Pain"] == 0.0
	assert row["Nausea"] == 1.0
	assert row["Ipsilateral_Rebound_Tenderness"] == 0.0


def test_build_input_row_uses_defaults_for_invalid_numeric(app_env: tuple[Any, TestClient]):
	app_module, _ = app_env
	form_data = {
		"age": "bad-value",
		"body_temperature": "",
		"wbc_count": "11.7",
		"crp": "NaN?",
		"neutrophil_percentage": "85.2",
		"appendix_diameter": "7.1",
		"lower_right_abd_pain": "yes",
		"migratory_pain": "yes",
		"nausea": "yes",
		"ipsilateral_rebound_tenderness": "no",
	}
	frame = app_module._build_input_row(form_data)
	row = frame.iloc[0].to_dict()

	assert row["Age"] == app_module._defaults["Age"]
	assert row["Body_Temperature"] == app_module._defaults["Body_Temperature"]
	assert row["CRP"] == app_module._defaults["CRP"]
	assert row["WBC_Count"] == 11.7
	assert row["Neutrophil_Percentage"] == 85.2


def test_session_token_roundtrip_and_expiration(
	monkeypatch: pytest.MonkeyPatch,
	app_env: tuple[Any, TestClient],
):
	app_module, _ = app_env
	monkeypatch.setattr(app_module.time, "time", lambda: 1_000)
	token = app_module._make_session_token("alice")

	monkeypatch.setattr(app_module.time, "time", lambda: 1_500)
	assert app_module._verify_session_token(token) == "alice"

	monkeypatch.setattr(app_module.time, "time", lambda: 5_000)
	assert app_module._verify_session_token(token) is None


def test_login_success_sets_session_cookie(client: TestClient):
	_login_as_admin(client)


def test_login_failure_returns_401_and_error_message(client: TestClient):
	response = client.post(
		"/login",
		data={"username": "admin", "password": "wrong"},
		follow_redirects=False,
	)
	assert response.status_code == 401
	assert "Identifiant ou mot de passe incorrect" in response.text


def test_form_requires_authentication_redirect(client: TestClient):
	response = client.get("/form", follow_redirects=False)
	assert response.status_code == 303
	assert response.headers.get("location") == "/login"


def test_predict_requires_authentication_redirect(client: TestClient):
	response = client.post("/predict", data=_valid_predict_payload(), follow_redirects=False)
	assert response.status_code == 303
	assert response.headers.get("location") == "/login"


def test_predict_authenticated_renders_probability_and_decision(client: TestClient):
	_login_as_admin(client)
	response = client.post("/predict", data=_valid_predict_payload(), follow_redirects=False)

	assert response.status_code == 200
	assert "Diagnostic Appendicite - Expert" in response.text
	assert "Prévisualisation & Analyse" in response.text


def test_api_predict_requires_authentication(client: TestClient):
	response = client.post("/api/predict", data=_valid_predict_payload())
	assert response.status_code == 401


def test_api_predict_and_history_flow_persists_record(client: TestClient):
	_login_as_admin(client)

	response = client.post("/api/predict", data=_valid_predict_payload())
	assert response.status_code == 200
	payload = response.json()
	assert payload["prob"] == "83.0"
	assert payload["risk_class"] == "danger"
	assert payload["shap_b64"] == "fake-shap"

	history = client.get("/api/history?limit=5")
	assert history.status_code == 200
	records = history.json()["records"]
	assert len(records) == 1
	assert records[0]["decision"] == "appendicite"
	assert records[0]["risk_class"] == "danger"
	assert records[0]["prob"] == 83.0

	record_id = records[0]["id"]
	by_id = client.get(f"/api/history?id={record_id}")
	assert by_id.status_code == 200
	one = by_id.json()["records"]
	assert len(one) == 1
	assert one[0]["id"] == record_id
