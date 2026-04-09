import os
from functools import lru_cache

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


@lru_cache(maxsize=1)
def _groq_client() -> Groq:
    key = os.getenv("GROQ_API_KEY")
    print(key, "999999999999999999999999999999999")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your environment or .env file for LLM warnings/recommendations."
        )
    return Groq(api_key=key)


def generate_llm_advice(department, incident_type, severity, risk_level):

    prompt = f"""
You are an industrial safety expert.

Department: {department}
Incident Type: {incident_type}
Severity: {severity}
Risk Level: {risk_level}

Respond strictly in this format:

WARNINGS:
- warning 1
- warning 2

RECOMMENDATIONS:
- recommendation 1
- recommendation 2
- recommendation 3
"""

    response = _groq_client().chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are an industrial safety advisor."},
            {"role": "user", "content": prompt}
        ]
    )

    text = response.choices[0].message.content.strip()

    warnings = []
    recommendations = []

    mode = None

    for line in text.split("\n"):
        line = line.strip()

        if line.lower().startswith("warnings"):
            mode = "warning"
            continue

        if line.lower().startswith("recommendations"):
            mode = "recommendation"
            continue

        if line.startswith("-"):
            if mode == "warning":
                warnings.append(line.replace("-", "").strip())

            elif mode == "recommendation":
                recommendations.append(line.replace("-", "").strip())

    return warnings, recommendations


