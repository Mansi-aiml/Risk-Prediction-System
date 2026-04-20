# from __future__ import annotations

# import json

# from django.http import JsonResponse
# from django.shortcuts import render
# from django.views.decorators.csrf import ensure_csrf_cookie
# from django.views.decorators.http import require_GET, require_POST

# from dashboard.services import get_processed_data, list_departments


# def _as_text_list(value) -> list[str]:
#     if value is None:
#         return []
#     if isinstance(value, (list, tuple)):
#         return [str(x) for x in value if x is not None and str(x).strip() != ""]
#     s = str(value).strip()
#     return [s] if s else []


# @ensure_csrf_cookie
# def index(request):
#     """GET: show form. POST: run prediction and render results (same flow as Streamlit)."""
#     context = {
#         "departments": [],
#         "load_error": None,
#         "risk": None,
#         "ts": None,
#         "ts_warning": None,
#         "prediction_error": None,
#         "forecast_days": 30,
#         "selected_department": "",
#         "distribution_bars": [],
#     }

#     try:
#         context["departments"] = list_departments()
#     except Exception as exc:
#         context["load_error"] = str(exc)
#         return render(request, "dashboard/index.html", context)

#     if request.method == "POST":
#         dept = (request.POST.get("department") or "").strip()
#         raw_days = request.POST.get("forecast_days")
#         try:
#             days = int(raw_days) if raw_days is not None and str(raw_days).strip() != "" else 30
#         except (TypeError, ValueError):
#             days = 30

#         context["selected_department"] = dept
#         context["forecast_days"] = days

#         if not dept:
#             context["prediction_error"] = "Please select a department."
#         elif days < 1 or days > 365:
#             context["prediction_error"] = "Forecast days must be between 1 and 365."
#         else:
#             from models.predictor import predict_future_risks
#             from models.ts_forecaster import run_ts_forecast

#             try:
#                 df_processed, last_training_date = get_processed_data()
#                 risk = predict_future_risks(dept, days, last_training_date)
#                 context["risk"] = risk
#                 context["warnings_list"] = _as_text_list(risk.get("warning"))
#                 context["recommendations_list"] = _as_text_list(risk.get("recommendation"))
#             except ValueError as exc:
#                 context["prediction_error"] = str(exc)
#             except Exception as exc:
#                 context["prediction_error"] = f"Prediction error: {exc}"

#             if context["risk"] is not None:
#                 try:
#                     ts_full = run_ts_forecast(
#                         df=df_processed,
#                         department=dept,
#                         forecast_days=days,
#                         last_training_date=last_training_date,
#                     )
#                     dist = ts_full.get("distribution") or {}
#                     max_v = max(dist.values(), default=0) or 1
#                     bars = []
#                     for k, v in dist.items():
#                         try:
#                             vi = int(v)
#                         except (TypeError, ValueError):
#                             vi = int(float(v))
#                         bars.append(
#                             {
#                                 "label": str(k),
#                                 "count": vi,
#                                 "pct": min(100.0, (float(v) / float(max_v)) * 100.0),
#                             }
#                         )
#                     context["ts"] = {
#                         "total_incidents": ts_full["total_incidents"],
#                         "trend": ts_full["trend"],
#                         "high_risk_week": ts_full["high_risk_week"],
#                     }
#                     context["distribution_bars"] = bars
#                 except Exception as exc:
#                     context["ts_warning"] = str(exc)

#     return render(request, "dashboard/index.html", context)


# @require_GET
# def api_departments(request):
#     try:
#         departments = list_departments()
#     except Exception as exc:
#         return JsonResponse({"error": str(exc)}, status=500)
#     return JsonResponse({"departments": departments})


# def _json_body(request) -> dict:
#     if not request.body:
#         return {}
#     return json.loads(request.body.decode("utf-8"))


# @require_POST
# def api_predict(request):
#     try:
#         payload = _json_body(request)
#     except json.JSONDecodeError:
#         return JsonResponse({"error": "Invalid JSON body"}, status=400)

#     department = (payload.get("department") or "").strip()
#     try:
#         forecast_days = int(payload.get("forecast_days", 30))
#     except (TypeError, ValueError):
#         return JsonResponse({"error": "forecast_days must be an integer"}, status=400)

#     if not department:
#         return JsonResponse({"error": "department is required"}, status=400)
#     if forecast_days < 1 or forecast_days > 365:
#         return JsonResponse({"error": "forecast_days must be between 1 and 365"}, status=400)

#     from models.predictor import predict_future_risks
#     from models.ts_forecaster import run_ts_forecast

#     try:
#         df_processed, last_training_date = get_processed_data()
#         risk = predict_future_risks(department,company, forecast_days, last_training_date)
#     except ValueError as exc:
#         return JsonResponse({"error": str(exc)}, status=400)
#     except Exception as exc:
#         return JsonResponse({"error": str(exc)}, status=500)

#     ts_summary = None
#     try:
#         ts_full = run_ts_forecast(
#             df=df_processed,
#             department=department,
              
#             forecast_days=forecast_days,
#             last_training_date=last_training_date,
#         )
#         ts_summary = {
#             "total_incidents": ts_full["total_incidents"],
#             "trend": ts_full["trend"],
#             "high_risk_week": ts_full["high_risk_week"],
#             "distribution": ts_full["distribution"],
#         }
#     except Exception as exc:
#         ts_summary = {"error": str(exc)}

#     return JsonResponse(
#         {"risk": risk, "time_series": ts_summary},
#         json_dumps_params={"default": str},
#     )

from __future__ import annotations

import json

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_POST

from dashboard.services import (
    get_processed_data,
    list_departments,
    list_companies,
    filter_data,
    resolve_effective_inputs,
)


# ─────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────
def _as_text_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value if x is not None and str(x).strip() != ""]
    s = str(value).strip()
    return [s] if s else []


# ─────────────────────────────────────────────────────────────
# UI VIEW (matches Streamlit behavior)
# ─────────────────────────────────────────────────────────────
@ensure_csrf_cookie
def index(request):
    context = {
        "departments": [],
        "companies": [],
        "load_error": None,
        "risk": None,
        "ts": None,
        "ts_warning": None,
        "prediction_error": None,
        "forecast_days": 30,
        "selected_department": "",
        "selected_company": "",
        "distribution_bars": [],
    }

    try:
        context["departments"] = list_departments()
        context["companies"] = list_companies()
    except Exception as exc:
        context["load_error"] = str(exc)
        return render(request, "dashboard/index.html", context)

    if request.method == "POST":
        dept = (request.POST.get("department") or "").strip()
        company = (request.POST.get("company") or "").strip()

        raw_days = request.POST.get("forecast_days")
        try:
            days = int(raw_days) if raw_days and str(raw_days).strip() != "" else 30
        except (TypeError, ValueError):
            days = 30

        context["selected_department"] = dept
        context["selected_company"] = company
        context["forecast_days"] = days

        if days < 1 or days > 365:
            context["prediction_error"] = "Forecast days must be between 1 and 365."
        else:
            from models.predictor import predict_future_risks
            from models.ts_forecaster import run_ts_forecast

            try:
                # 🔹 FILTER DATA (same as Streamlit)
                df_filtered = filter_data(dept or None, company or None)

                if df_filtered.empty:
                    context["prediction_error"] = "No data available for this selection."
                else:
                    # 🔹 RESOLVE EFFECTIVE INPUTS
                    dept, company = resolve_effective_inputs(
                        df_filtered, dept or None, company or None
                    )

                    df_processed, last_training_date = get_processed_data()

                    # 🔹 ML PREDICTION
                    risk = predict_future_risks(
                        dept,
                        company,
                        days,
                        last_training_date
                    )

                    context["risk"] = risk
                    context["warnings_list"] = _as_text_list(risk.get("warning"))
                    context["recommendations_list"] = _as_text_list(risk.get("recommendation"))

                    # 🔹 TIME SERIES (uses filtered data)
                    try:
                        ts_full = run_ts_forecast(
                            df=df_filtered,
                            department=dept,
                            forecast_days=days,
                            last_training_date=last_training_date,
                        )

                        dist = ts_full.get("distribution") or {}
                        max_v = max(dist.values(), default=0) or 1

                        bars = []
                        for k, v in dist.items():
                            try:
                                vi = int(v)
                            except (TypeError, ValueError):
                                vi = int(float(v))

                            bars.append({
                                "label": str(k),
                                "count": vi,
                                "pct": min(100.0, (float(v) / float(max_v)) * 100.0),
                            })

                        context["ts"] = {
                            "total_incidents": ts_full["total_incidents"],
                            "trend": ts_full["trend"],
                            "high_risk_week": ts_full["high_risk_week"],
                        }

                        context["distribution_bars"] = bars

                    except Exception as exc:
                        context["ts_warning"] = str(exc)

            except Exception as exc:
                context["prediction_error"] = f"Prediction error: {exc}"

    return render(request, "dashboard/index.html", context)


# ─────────────────────────────────────────────────────────────
# APIs
# ─────────────────────────────────────────────────────────────
@require_GET
def api_departments(request):
    try:
        return JsonResponse({"departments": list_departments()})
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)


@require_GET
def api_companies(request):
    try:
        return JsonResponse({"companies": list_companies()})
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)


def _json_body(request) -> dict:
    if not request.body:
        return {}
    return json.loads(request.body.decode("utf-8"))


@require_POST
def api_predict(request):
    try:
        payload = _json_body(request)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    department = (payload.get("department") or "").strip()
    company = (payload.get("company") or "").strip()

    try:
        forecast_days = int(payload.get("forecast_days", 30))
    except (TypeError, ValueError):
        return JsonResponse({"error": "forecast_days must be an integer"}, status=400)

    if forecast_days < 1 or forecast_days > 365:
        return JsonResponse({"error": "forecast_days must be between 1 and 365"}, status=400)

    from models.predictor import predict_future_risks
    from models.ts_forecaster import run_ts_forecast

    try:
        df_filtered = filter_data(department or None, company or None)

        if df_filtered.empty:
            return JsonResponse({"error": "No data available"}, status=400)

        department, company = resolve_effective_inputs(
            df_filtered,
            department or None,
            company or None
        )

        df_processed, last_training_date = get_processed_data()

        # 🔹 ML Prediction
        risk = predict_future_risks(
            department,
            company,
            forecast_days,
            last_training_date
        )

    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)

    # 🔹 Time Series
    try:
        ts_full = run_ts_forecast(
            df=df_filtered,
            department=department,
            forecast_days=forecast_days,
            last_training_date=last_training_date,
        )

        ts_summary = {
            "total_incidents": ts_full["total_incidents"],
            "trend": ts_full["trend"],
            "high_risk_week": ts_full["high_risk_week"],
            "distribution": ts_full["distribution"],
        }

    except Exception as exc:
        ts_summary = {"error": str(exc)}

    return JsonResponse(
        {
            "risk": risk,
            "time_series": ts_summary
        },
        json_dumps_params={"default": str},
    )
