
"""
Models package initialization.
File: src/models/__init__.py

Imports all model classes for easy access:
from models import DriverState, CascadingInsertion, WeeklySchedule
"""

# Import all classes from individual modules
from .driver_state import (
    DriverState, 
    DailyAssignment,
    DAILY_DUTY_LIMIT_MIN,
    STANDARD_REST_MIN,
    EMERGENCY_REST_MIN,
    WEEKEND_REST_MIN,
    MAX_EMERGENCY_PER_WEEK,
    MAX_DELAY_TOLERANCE_MIN
)

from .cascading_insertion import (
    CascadingInsertion,
    InsertionStep,
    InsertionType
)

from .weekly_schedule import (
    WeeklySchedule,
    WeekendBreak
)

# Make all classes available at package level
__all__ = [
    # Core classes
    'DriverState',
    'DailyAssignment', 
    'CascadingInsertion',
    'InsertionStep',
    'InsertionType',
    'WeeklySchedule',
    'WeekendBreak',
    
    # Constants
    'DAILY_DUTY_LIMIT_MIN',
    'STANDARD_REST_MIN', 
    'EMERGENCY_REST_MIN',
    'WEEKEND_REST_MIN',
    'MAX_EMERGENCY_PER_WEEK',
    'MAX_DELAY_TOLERANCE_MIN'
]