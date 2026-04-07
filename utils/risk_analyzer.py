from config.settings import RISK_LEVELS

# ─── Incident-type → recommended action ───────────────────────────────────────
_RECOMMENDATIONS: dict[str, str] = {
    "electrical":       "Inspect all wiring systems, check insulation integrity, and enforce LOTO procedures.",
    "fire":             "Test fire suppression systems, conduct evacuation drills, and verify extinguisher placement.",
    "chemical":         "Review SDS compliance, inspect PPE availability, and audit chemical storage segregation.",
    "spill":            "Activate spill response protocol, inspect secondary containment, and update MSDS records.",
    "fall":             "Inspect elevated walkways, mandate safety harness usage, and apply anti-slip flooring.",
    "slip":             "Inspect floor surfaces, improve drainage, and enforce appropriate footwear policy.",
    "mechanical":       "Schedule preventive maintenance, verify machine guarding, and audit LOTO compliance.",
    "ergonomic":        "Assess workstation setups, enforce proper manual-handling techniques, and schedule breaks.",
    "struck":           "Enforce PPE usage (hard hats), establish exclusion zones in active work areas.",
    "caught":           "Audit machine guarding, enforce lockout/tagout procedures, and restrict access zones.",
    "heat":             "Deploy cooling stations, monitor shift durations, and enforce mandatory hydration breaks.",
    "explosion":        "Audit pressurised systems, review HAZOP findings, and inspect all pressure relief valves.",
    "noise":            "Conduct audiometric testing, supply hearing protection, and reduce source noise levels.",
    "radiation":        "Ensure shielding compliance, enforce dosimeter use, and restrict unauthorised access.",
    "biological":       "Enforce hygiene protocols, provide appropriate PPE, and conduct health surveillance.",
    "vehicle":          "Review traffic management plan, enforce speed limits, and conduct driver safety training.",
    "default":          "Conduct a comprehensive safety audit and reinforce general safety training protocols.",
}


def get_risk_level(severity_type: str) -> str:
    """
    Map a severity label to a categorical risk level.

    Returns one of: CRITICAL | HIGH | MEDIUM | LOW
    Falls back to MEDIUM if no keyword matches.
    """
    lower = severity_type.lower()
    for level, keywords in RISK_LEVELS.items():
        if any(kw.lower() in lower for kw in keywords):
            return level
    return "MEDIUM"


def get_recommendation(incident_type: str) -> str:
    """
    Return a recommended safety action for a given incident type.

    Performs keyword matching (case-insensitive) against the lookup table.
    """
    lower = incident_type.lower()
    for key, action in _RECOMMENDATIONS.items():
        if key in lower:
            return action
    return _RECOMMENDATIONS["default"]


def get_warning(incident_type: str, risk_level: str) -> str:
    """
    Compose a contextual warning message combining risk level severity
    and the specific incident type identified.
    """
    level_messages = {
        "CRITICAL": (
            "CRITICAL ALERT: Immediate intervention required — life-threatening risk detected."
        ),
        "HIGH": (
            "HIGH RISK: Initiate urgent safety inspection and notify the safety officer immediately."
        ),
        "MEDIUM": (
            "ELEVATED RISK: Schedule a targeted safety audit and reinforce relevant protocols."
        ),
        "LOW": (
            "ROUTINE RISK: Ensure all standard safety procedures are followed as per SOP."
        ),
    }
    base = level_messages.get(risk_level, level_messages["MEDIUM"])
    return f"{base} Specific concern: {incident_type}."
