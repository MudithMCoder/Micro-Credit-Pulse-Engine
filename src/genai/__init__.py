"""GenAI Loan Officer Advisor module (Module 5).

Generates plain-language loan officer advisory summaries using LLM + SHAP drivers.
"""
from .advisor import generate_advisories, generate_portfolio_report

__all__ = ["generate_advisories", "generate_portfolio_report"]
