"""Pydantic models for API request validation."""

from pydantic import BaseModel


class Base(BaseModel):
    """Base model with core customer features for validation."""
    age: int
    tenure_months: int
    monthly_charges: float
    contract_type: int
    support_tickets: int


class ChurnInput(Base):
    """Request model for churn prediction endpoint."""
    pass